import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import MulticlassCalibrationError
import numpy as np
from math import pi




def calculate_log_posterior(particle_ensemble, batch_X, batch_y, config, prior):
    # calculate Negative log posterior
    # print('calculating log posterior')

    # assuming prior to be standard gaussian
    batch_size = batch_X.shape[0]
    
    pred_y = particle_ensemble(batch_X)


    pred_y = pred_y.reshape(particle_ensemble.num_particles, batch_size, config['task_dim'])

    # print('pred_y:',pred_y.shape)
    # print('batch_y.expand(pred_y.shape).shape:',batch_y.expand(pred_y.shape).shape)
    # print('len ensemble:',particle_ensemble.num_particles)

    if not config['is_classification']:
        # Log likelihood for regression (Gaussian)
        
        log_likelihoods = 0.5 * torch.mean((batch_y.expand(pred_y.shape) - pred_y) ** 2, dim=1)
        log_likelihoods = log_likelihoods*batch_size/config['likelihood_std']**2

    else:  # classification
        # Log likelihood for classification (categorical cross-entropy)
        log_likelihoods = torch.stack([F.cross_entropy(pred_y[i], batch_y.squeeze(1).long()) for i in range(pred_y.size(0))])
        # print('log_likelihoods.shape:',log_likelihoods.shape)
        log_likelihoods = log_likelihoods*batch_size
        

        # print('log_likelihoods for classification',log_likelihoods)

    # Log prior (assuming standard Gaussian)
    # log_priors = 0.5 * sum(torch.sum(p**2) for p in particle_ensemble.parameters())
    # log_priors = prior.log
    
    # log_priors = sum(prior.log_prob(p).sum() for p in particle_ensemble.parameters())
    # print(log_priors)
    
    # Negative log posterior (our optimization objective)
    # log_likelihoods = -log_likelihoods*batch_size/config['likelihood_std']**2

    # losses = -(log_likelihoods + log_priors)
    
    # log_posteriors = -log_likelihoods
    log_posteriors = -log_likelihoods
    # print('log_likelihoods:',log_likelihoods)

    return log_posteriors



def compute_metrics(data_loader, model, device, config):
    model.eval()
    model.to(device)
    
    if config['is_classification']:
        all_logits = []
        all_probabilities = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # Get outputs from ensemble models
                outputs = torch.stack([m(batch_x) for m in model.models])  # Shape: (num_particles, batch_size, task_dim)

                # Align shapes
                pred_y = outputs.reshape(model.num_particles, batch_x.size(0), config['task_dim'])
                batch_y_expanded = batch_y.expand(pred_y.shape)

                # Compute mean logits over ensemble
                mean_logits = pred_y.mean(dim=0)  # Shape: (batch_size, num_classes)

                # Compute probabilities
                mean_probabilities = F.softmax(mean_logits, dim=-1)  # Shape: (batch_size, num_classes)

                all_logits.append(mean_logits.cpu())
                all_probabilities.append(mean_probabilities.cpu())
                all_targets.append(batch_y.squeeze().cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_probabilities = torch.cat(all_probabilities, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # print('all_logits.shape',all_logits.shape)
        # print('all_probabilities.shape',all_probabilities.shape)
        # print('all_targets.shape',all_targets.shape)

        # Predicted labels
        predicted_labels = all_probabilities.argmax(dim=-1)

        # Accuracy
        accuracy = (predicted_labels == all_targets.long()).float().mean().item()

        # Cross Entropy Loss
        cross_entropy_loss = F.cross_entropy(all_logits, all_targets.long()).item()

        # Negative Log Likelihood
        nll_loss = F.nll_loss(F.log_softmax(all_logits, dim=-1), all_targets.long()).item()

        # Brier Score
        num_classes = all_probabilities.size(-1)
        true_labels_one_hot = F.one_hot(all_targets.long(), num_classes=num_classes).float()
        brier_score = ((all_probabilities - true_labels_one_hot) ** 2).sum(dim=1).mean().item()

        # Entropy
        entropy = (-all_probabilities * torch.log(all_probabilities + 1e-12)).sum(dim=1).mean().item()

        # AUROC (using sklearn)
        try:
            auroc = roc_auc_score(all_targets.numpy(), all_probabilities.numpy()[:, 1])
        except ValueError:
            auroc = None  # Handle the case when AUROC cannot be computed
            auroc=-1 
            

        # ECE (Expected Calibration Error)
        # You can implement ECE calculation here or use existing libraries like torchmetrics
        ECE_loss = MulticlassCalibrationError(num_classes=config['task_dim'], n_bins=15, norm='l1')
        ece_value = ECE_loss(all_probabilities, all_targets).item()


        metrics = {
            'Accuracy': accuracy,
            'Cross Entropy': cross_entropy_loss,
            'NLL': nll_loss,
            'Brier': brier_score,
            'Entropy': entropy,
            'AUROC': auroc,
            'ECE': ece_value  # Add ECE calculation if implemented
        }



    else: # regression
        all_predictions = []
        all_variances = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # Get outputs from ensemble models
                outputs = torch.stack([m(batch_x) for m in model.models])  # Shape: (num_particles, batch_size, task_dim)

                # Align shapes
                pred_y = outputs.reshape(model.num_particles, batch_x.size(0), config['task_dim'])
                batch_y_expanded = batch_y.expand(pred_y.shape)

                # Mean predictions over ensemble
                mean_predictions = pred_y.mean(dim=0)  # Shape: (batch_size, 1)

                # Variance over ensemble
                variance = pred_y.var(dim=0) + 1e-12  # Shape: (batch_size, 1), constant for stability

                all_predictions.append(mean_predictions.cpu())
                all_variances.append(variance.cpu())
                all_targets.append(batch_y.squeeze().cpu())


        all_predictions = torch.cat(all_predictions, dim=0)
        all_variances = torch.cat(all_variances, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # print('predictions shape',all_predictions.squeeze().shape)
        # print('target shape',all_targets.shape)

        # MSE
        
        mse = F.mse_loss(all_predictions.squeeze(), all_targets).item()
        # mse = nn.MSELoss()(all_predictions.squeeze(), all_targets).item()

        # RMSE
        rmse = torch.sqrt(F.mse_loss(all_predictions.squeeze(), all_targets)).item()

        # Negative Log Likelihood
        nll_loss = F.gaussian_nll_loss(
            all_predictions.squeeze(),
            all_targets,
            all_variances.squeeze(),
            eps=1e-12  # Small constant to avoid numerical issues
        ).item()

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'NLL': nll_loss
        }

    return metrics


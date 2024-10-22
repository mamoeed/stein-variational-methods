"""
This is the main file which runs the experiments and reports results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
import torch.autograd as autograd

from utils.dataset import Dataset
from torch.utils.data import DataLoader
from utils.kernel import RBF

import numpy as np, pandas as pd
import os, sys
import yaml
from tqdm import tqdm
import copy

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, train_test_split

from model_architectures import NeuralNetworkEnsemble
from utils.loss import *
from methods import *



def run_experiment(config, seed):

    # load complete data as pandas dataframe
    # separate into train/val/test
    # 

    ############## wine data
    if config['task'] == 'wine':
        file_path =  'data/wine/winequality-red.csv' 
        df = pd.read_csv(file_path, sep=';')
        print(df.head())
        X = df.drop(columns=['quality']).values
        y = df['quality'].values

    ############## yacht data
    if config['task'] == 'yacht':
        file_path =  'data/yacht/yachthydrodynamics.csv' 
        data = pd.read_csv(file_path, sep=';')

        print(data.head())
        # Split the data into features and target variable
        X = data.iloc[:, :-1].values  # Features
        y = data.iloc[:, -1].values   # Target variable
        
    ############## power data 
    if config['task'] == 'power':
        file_path =  'data/power/Folds5x2_pp.xlsx'
        data = pd.read_excel(file_path)
        print(data.head())

        #PE is target variable
        X = data.iloc[:, :-1].values  # Features
        y = data.iloc[:, -1].values   # Target variable

    ############## australian data
    if config['task'] == 'australian':
        file_path =  'data/australian/australian_data.xlsx'
        data = pd.read_excel(file_path)
        print(data.head())
        print("Dataset length:", len(data))
        print("Number of columns before one-hot encoding:", len(data.columns))
        unique_classes = data['Class'].unique()
        n_classes = len(unique_classes)  # Count of unique classes
        print("Unique status classes:", unique_classes)
        print("Number of unique classes:", n_classes)
        # One-hot encode the "status" column and update the dataframe
        encoder = OneHotEncoder(sparse_output=False, drop='first')  # Avoid multicollinearity by dropping first
        data_encoded = data.copy()  # Create a copy to keep the original data intact
        # are input variables NOT one hot encoded?
        data_encoded['Class'] = encoder.fit_transform(data[['Class']]).ravel()  # Flatten array and assign
        # Separating features and target variable
        X = data_encoded.drop(columns=['Class']).values  # Features: all columns except 'status'
        y = data_encoded['Class'].values  # Target variable: encoded 'status'

    ############## breast data
    if config['task'] == 'breast':
        file_path =  'data/breast/wisconsin_breast_cancer_data.xlsx'
        data = pd.read_excel(file_path)
        #data = pd.read_excel(file_path)
        print(data.head())
        
        print("Dataset length:", len(data))
        print("Number of columns before one-hot encoding:", len(data.columns))

        unique_classes = data['Diagnosis'].unique()
        n_classes = len(unique_classes)  # Count of unique classes
        print("Unique status classes:", unique_classes)
        print("Number of unique classes:", n_classes)

        #assert n_classes == config.task.dim_problem
        # One-hot encode the "status" column and update the dataframe
        encoder = OneHotEncoder(sparse_output=False, drop='first')  # Avoid multicollinearity by dropping first
        data_encoded = data.copy()  # Create a copy to keep the original data intact
        data_encoded['Diagnosis'] = encoder.fit_transform(data[['Diagnosis']]).ravel()  # Flatten array and assign

        # Separating features and target variable
        X = data_encoded.drop(columns=['Diagnosis']).values  # Features: all columns except 'status'
        y = data_encoded['Diagnosis'].values  # Target variable: encoded 'status'

    ############### MNIST 
    if config['task'] == 'mnist':
        # print('not implemented')
        file_path =  'data/mnist/mnist_data.csv'
        
        # if config.task.n_samples == True:
        data = pd.read_csv(file_path) # pick all 60K samples
        # else:
            # data = pd.read_csv(file_path, nrows=config.task.n_samples)
        print('hello world')
        # Assuming the last column in the DataFrame is 'label'
        X = data.iloc[:, :-1].values  # Features (pixel values)
        y = data.iloc[:, -1].values   # Labels
        print('############################### \nX.shape\n',X.shape)
        # return 0
        # Normalize pixel values to be between 0 and 1
        X = X / 255.0

        # Reshape the images from 784 pixels to 28x28 (if necessary, depending on model input requirement)
        # Adding a channel dimension for grayscale
        X_reshaped = X.reshape(-1, 28, 28)  # N, C, H, W format for PyTorch Conv2D
        # Convert labels and features into torch tensors
        # X = torch.tensor(X_reshaped, dtype=torch.float32)
        # y = torch.tensor(y, dtype=torch.long)
        X = X_reshaped
        

            

    ################## data handling and spliting

    print('data loaded successfully')

    torch.manual_seed(seed)
    # random_seed_data_split = 1
    
    # separate out test first:
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2) # , random_state=random_seed_data_split

    # standardize data
    if config['task'] != 'mnist':
        scaler = StandardScaler()
        X_train_val = scaler.fit_transform(X_train_val)
        X_test = scaler.transform(X_test)
        

    # separate train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25) # , random_state=random_seed_data_split
    
    # train/val/test ==> 60/20/20 of total data size

    print('data split into train/val/test')
    print('X_train.shape:', X_train.shape)
    print('y_train.shape:', y_train.shape)
    print('X_val.shape', X_val.shape)
    print('y_val.shape', y_val.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)

    print('Number of particles:',config['num_particles'])

    
    


    # make dataloaders:
    train_dataset = Dataset(X_train, y_train)
    val_dataset = Dataset(X_val, y_val)
    test_dataset = Dataset(X_test, y_test)

    print("length train", len(train_dataset))
    print("length val", len(val_dataset))
    print("length test", len(test_dataset))
    
    #create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size= config['batch_size'], shuffle=True) # , shuffle=True
    val_dataloader = DataLoader(val_dataset, batch_size= config['batch_size']) # , shuffle=True
    test_dataloader = DataLoader(test_dataset, batch_size= config['batch_size']) # , shuffle=True

    
    # initialize model, only MLP for now
    input_dim = X_train.shape[1]  # Number of features in x_train
    
    if config['task'] == 'mnist':
        particle_ensemble = LeNetEnsemble(config['task_dim'],config['num_particles'])
    else:
        particle_ensemble = NeuralNetworkEnsemble(input_dim,config['task_dim'],config['num_particles'],50) 

    print('particle_ensemble.models',particle_ensemble.models)
    # print('ensemble initialised with parameters:', [p for p in particle_ensemble.parameters()])
    # for p in particle_ensemble.parameters():
        # print('the parameters are:')
        # print(p)
    # for model in particle_ensemble.parameters():
    #     print(model.shape)
    # print((particle_ensemble[0]))

    
    
    train_epoch_metrics = []
    val_epoch_metrics = []
    ################ training model now
    num_epochs = config['num_epochs']
    
    metrics_evaluation = {} # contain all relevant metrics for classification/regression tasks that need to be tracked

    
    if config['method'] == 'SGD_ensembles':
        optimizer = SGD_ensembles(params = particle_ensemble.parameters(), lr = config['lr'])
    elif config['method'] == 'SGLD':
        optimizer = SGLD(params = particle_ensemble.parameters(), lr = config['lr'])
    elif config['method'] == 'SVGD':
        print('RUNNING ',config['method'])
        optimizer = SVGD(particle_ensemble, lr = config['lr'], gamma=1e-4, annealing=config['anneal'], annealing_period=num_epochs*len(train_dataloader)) # 1e-4 for wine

    elif config['method'] == 'sSVGD':
        optimizer = SVGD(particle_ensemble, lr = config['lr'], gamma=1e-4, stochastic=True, annealing=config['anneal'], annealing_period=num_epochs*len(train_dataloader)) # 1e-3 for wine

    elif config['method'] == 'SVN_laplace':
        optimizer = SVN_laplace(particle_ensemble, lr = config['lr'], gamma=1e-4,  config=config, exponential_factor=1, annealing=config['anneal'], annealing_period=num_epochs*len(train_dataloader)) # wine 1e-4
    
    elif config['method'] == 'sSVN_laplace':
        optimizer = SVN_laplace(particle_ensemble, lr = config['lr'], gamma=1e-4,  config=config, exponential_factor=1,stochastic=True, annealing=config['anneal'], annealing_period=num_epochs*len(train_dataloader))
    
    elif config['method'] == 'SVN_adam':
        optimizer = SVN_adam(particle_ensemble, lr = 1e-7, gamma=1e-8 , config=config)

    elif config['method'] == 'sSVN_adam':
        optimizer = SVN_adam(particle_ensemble, lr = 1e-7, gamma=1e-8 , config=config,stochastic=True)

    else:
        print('not implemented method',config['method'])


    #Early Stopping and loading best model based on validation error
    best_metric = float('inf')
    best_epoch = -1  # To track the epoch number of the best loss metric
    epochs_since_improvement = 0
    best_modellist_state = None
    patience = 5  # Number of epochs to wait before reducing learning rate

    train_metrics = compute_metrics(train_dataloader, particle_ensemble, 'cpu',config)
    train_epoch_metrics.append(train_metrics)
    val_metrics = compute_metrics(val_dataloader, particle_ensemble, 'cpu',config)
    val_epoch_metrics.append(val_metrics)

    print('before training metrics: ',train_metrics)

    for epoch in range(num_epochs):
        
        print(f'epoch: {epoch+1}/{num_epochs}')

        for step, (batch_X, batch_y) in enumerate(tqdm(train_dataloader)):
            # one step is one gradient update of all particles

            # print('step:',step,'\t batch_X:',batch_X.shape, '\t batch_y:',batch_y.shape)

            optimizer.zero_grad()

            ############ log_posterior and its gradient
            prior = torch.distributions.Normal(0, 1)
            log_posterior = calculate_log_posterior(particle_ensemble, batch_X, batch_y, config, prior).sum()
            # print('log_posterior:',log_posterior)
            # print('log_posterior:',log_posterior)
            log_posterior.backward()
            # print(log_posterior.item())
            # grad_log_posterior = autograd.grad(log_posterior, particle_ensemble.parameters())
            
            # print('######################################\ngrad_log_posterior.shape: ')
            # for p in particle_ensemble.parameters():
            #     print(p.grad.shape == p.shape)

            ############ RBF kernel gram matrix and its gradient




            ########### optimizer update step
            if config['method'] in ["SVN_laplace","sSVN_laplace","SVN_adam","sSVN_adam"]:
                optimizer.step((batch_X, batch_y))
            else:
                optimizer.step() 

            # scheduler.step()
        
        # one epoch completed. compute metrics for evaluation of model
        train_metrics = compute_metrics(train_dataloader, particle_ensemble, 'cpu',config)
        val_metrics = compute_metrics(val_dataloader, particle_ensemble, 'cpu',config)
        train_epoch_metrics.append(train_metrics)
        val_epoch_metrics.append(val_metrics)
        print('train metrics: ',train_metrics)
        print('val loss metrics: ',val_metrics)

        # keep track of best performing model on validation data
        # if config['is_classification']:
        #     model_selection_metric = val_metrics[config['model_selection_metric']]
        #     if model_selection_metric < best_metric:
        #         best_metric = model_selection_metric
        #         epochs_since_improvement = 0
        #         best_epoch = epoch
        #         best_particles_state = copy.deepcopy(particle_ensemble)  # Deep copy to save the particles state
                
        #     else:
        #         epochs_since_improvement += 1
        #         print(f"{config['model_selection_metric']} did not improve in this epoch :(")
                
        # else: # regression
        model_selection_metric = val_metrics[config['model_selection_metric']]
        if model_selection_metric < best_metric:
            best_metric = model_selection_metric
            epochs_since_improvement = 0
            best_epoch = epoch
            best_particles_state = copy.deepcopy(particle_ensemble)  # Deep copy to save the particles state
        
        else:
            epochs_since_improvement += 1
            print(f"{config['model_selection_metric']} did not improve in this epoch :(")

            # Check if we should reduce the learning rate
            if epochs_since_improvement >= patience:
                optimizer.reduce_lr()
                print(f"Reducing learning rate to {optimizer.get_lr()}")
                epochs_since_improvement = 0  # Reset the counter
        
        print(f"Current learning rate: {optimizer.get_lr()}")
            


        



        
        # print(calculate_metrics('classification', particle_ensemble, train_dataloader, 3, 2))

    print('Train Negative Log Likelihood trend : ',[round(i['NLL'],6) for i in train_epoch_metrics])
    print('Val Negative Log Likelihood trend : ',[round(i['NLL'],6) for i in val_epoch_metrics])

    
        

        
    
    # pick model with best val MSE/NLL and compute performance on test set
    test_metrics = compute_metrics(test_dataloader, best_particles_state, 'cpu',config)
    print('PERFORMANCE ON TEST SET:\n',test_metrics)


    # plot train/val curves and metrics on test

    if config['is_classification']:
        print('Val cross entropy trend : ',[round(i['Cross Entropy'],6) for i in val_epoch_metrics])
        print('Val Brier trend : ',[round(i['Brier'],6) for i in val_epoch_metrics])
        print('Val Entropy trend : ',[round(i['Entropy'],6) for i in val_epoch_metrics])
        print('Val AUROC trend : ',[round(i['AUROC'],6) for i in val_epoch_metrics])
        print('Val ECE trend : ',[round(i['ECE'],6) for i in val_epoch_metrics])

        test_metrics['train_NLL'] = [round(i['NLL'],6) for i in train_epoch_metrics]
        test_metrics['train_cross_entropy'] = [round(i['Cross Entropy'],6) for i in train_epoch_metrics]
        test_metrics['val_NLL'] = [round(i['NLL'],6) for i in val_epoch_metrics]
        test_metrics['val_cross_entropy'] = [round(i['Cross Entropy'],6) for i in val_epoch_metrics]
        

    else:
        print('Val MSE trend : ',[round(i['MSE'],6) for i in val_epoch_metrics])
        print('Val RMSE trend : ',[round(i['RMSE'],6) for i in val_epoch_metrics])
        test_metrics['train_NLL'] = [round(i['NLL'],6) for i in train_epoch_metrics]
        test_metrics['train_MSE'] = [round(i['MSE'],6) for i in train_epoch_metrics]
        test_metrics['val_NLL'] = [round(i['NLL'],6) for i in val_epoch_metrics]
        test_metrics['val_MSE'] = [round(i['MSE'],6) for i in val_epoch_metrics]

    




    return test_metrics












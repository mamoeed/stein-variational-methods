"""
Particle based Methods which are coded as pytorch optimizers.
Ensembles 
SGLD
SVGD
SVN
sSVN
sSVGD
"""


import torch
import torch.optim as optim
from model_architectures import *
from sklearn.metrics.pairwise import rbf_kernel
from utils.dataset import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import scipy
import math
from utils.stein_utils import *


from laplace import  DiagLaplace
from laplace.curvature import AsdlGGN, AsdlEF, GGNInterface
from laplace import Laplace





class SGD_ensembles(optim.Optimizer):
    # same as torch.optim.Optimizer.SGD
    
    def __init__(self, params, lr=0.01, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SGD_ensembles, self).__init__(params, defaults)

        # New attributes for learning rate schedule
        self.initial_lr = torch.tensor(float(lr))
        self.current_lr = torch.tensor(float(lr))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            with torch.no_grad():
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data

                    if group['weight_decay'] != 0:
                        d_p.add_(group['weight_decay'], p.data)

                    p.data.add_(torch.tensor(float(group['lr']), dtype=torch.float64), d_p)
                    # p.data.add_(d_p, alpha=group['lr'])

        return loss
    def reduce_lr(self):
        """Reduce the learning rate by half."""
        self.current_lr *= 1.
        for param_group in self.param_groups:
            param_group['lr'] = torch.tensor(float(self.current_lr), dtype=torch.float64)
        print(f"\n\n LEARNING RATE REDUCED TO:{self.current_lr} \n\n")

    def get_lr(self):
        """Get the current learning rate."""
        return self.current_lr
    

class SGLD(optim.Optimizer):
    # stochastic gradient langevin descent
    
    def __init__(self, params, lr=0.01, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SGLD, self).__init__(params, defaults)

        # New attributes for learning rate schedule
        self.initial_lr = torch.tensor(float(lr))
        self.current_lr = torch.tensor(float(lr))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            with torch.no_grad():
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data

                    if group['weight_decay'] != 0:
                        d_p.add_(group['weight_decay'], p.data)

                    noise = torch.normal(mean=0, std=torch.sqrt(torch.tensor(float(group['lr']), dtype=torch.float64)), size=p.size(), device=p.device)

                    p.data.add_(torch.tensor(float(group['lr']), dtype=torch.float64), d_p)
                    p.data.add_(noise)

        return loss
    
    def reduce_lr(self):
        """Reduce the learning rate by half."""
        self.current_lr *= 1.
        for param_group in self.param_groups:
            param_group['lr'] = torch.tensor(float(self.current_lr), dtype=torch.float64)
        print(f"\n\n LEARNING RATE REDUCED TO:{self.current_lr} \n\n")

    def get_lr(self):
        """Get the current learning rate."""
        return self.current_lr


class SVGD(optim.Optimizer):
    def __init__(self, particle_ensemble, lr=1e-3,  gamma=-1, stochastic = False, annealing=False, annealing_period=500, annealing_min=0.0, annealing_max=1.0):
        # if not isinstance(params, NeuralNetworkEnsemble):
        #     raise TypeError("params argument should be a NeuralNetworkEnsemble")
        self.particle_ensemble = particle_ensemble
        
        self.stochastic = stochastic # add langevin noise to the gradients transforming into MCMC

        # Create a single param group for all particles,  TODO: why is this?????? 
        param_group = {'params': self.particle_ensemble.parameters(), 'lr': torch.tensor(float(lr), dtype=torch.float64)}

        
        defaults = dict(lr=torch.tensor(float(lr), dtype=torch.float64), gamma=gamma)
        super(SVGD, self).__init__(self.particle_ensemble.parameters(), defaults)

        # annealing parameters
        self.annealing = annealing # bool flag for annealing 
        self.annealing_period = annealing_period # total duration (= num_epochs*batch_size for hyperbolic tanget annealing schedule)
        self.annealing_min = annealing_min
        self.annealing_max = annealing_max
        self.current_step = 0
        self.annealed_param = self.annealing_min # parameter that is annealed

        # New attributes for learning rate schedule
        self.initial_lr = torch.tensor(float(lr))
        self.current_lr = torch.tensor(float(lr))


    @torch.no_grad()
    def step(self):
        # Get flattened parameters for each particle
        particles = []
        particles_grad = []
        for model in self.particle_ensemble.models:
            particle_params = torch.cat([p.view(-1) for p in model.parameters()])
            particles.append(particle_params)
            particle_grad_params = torch.cat([p.grad.view(-1) for p in model.parameters()])
            particles_grad.append(particle_grad_params)

        particles = torch.stack(particles)  # Shape: (N, D) where N is number of particles and D is number of parameters
        particles_grad = torch.stack(particles_grad) # Shape: (N, D) gradient of params of each particles flattened


        # print('shape of particles variable:',particles.shape)
        # print('shape of particles_grad variable:',particles_grad.shape)

        # Compute pairwise distances
        dist = torch.cdist(particles, particles)
        # print('pairwise distances shape:',dist.shape)
        
        # Convert to numpy for scikit-learn
        particles_np = particles.cpu().numpy()

        # Compute gamma if not provided
        gamma = self.defaults['gamma']
        if gamma < 0:
            gamma = 1.0 / particles_np.shape[1]

        # Compute kernel matrix using scikit-learn
        K = rbf_kernel(particles_np, gamma=gamma)

        # Convert back to PyTorch tensor
        K = torch.tensor(K, dtype=particles.dtype, device=particles.device)

        # Compute gradient of kernel
        grad_K = -K.unsqueeze(-1) * (particles.unsqueeze(1) - particles.unsqueeze(0))

        # print('K',K, 'shape:',K.shape) # NxN
        # print('grad_K',grad_K, 'shape:',grad_K.shape) # NxNxD
        
        # print(K)

        if self.annealing:
            v_svgd =  self.annealed_param*(torch.einsum('mn, mo -> no', K, particles_grad)/K.shape[0]) - (torch.mean(grad_K, dim=0)) # shape = N x D ==> num_particle x num_parameters_per_model
        else:
            v_svgd =  (torch.einsum('mn, mo -> no', K, particles_grad)/K.shape[0] ) - (torch.mean(grad_K, dim=0)) # shape = N x D ==> num_particle x num_parameters_per_model

        

        # update parameters below
        for model, grads in zip(self.particle_ensemble.models, v_svgd):
            flat_params = torch.cat([p.view(-1) for p in model.parameters()])
            
            start = 0  # Start index for slicing gradients from grads
            for param in model.parameters():
                # Number of elements in the current parameter
                num_param_elements = param.numel()
                end = start + num_param_elements

                # Slice the gradient for the current parameter
                grad_for_param = grads[start:end].view(param.size())
                
                # update the parameters 
                if param.grad is None:
                    print('grad is none LOLLL')
                    continue
                else:
                    param.data.add_(self.param_groups[0]['lr'],grad_for_param)
                    # param.data.add_(self.current_lr,grad_for_param)
                
                # Update the start index for the next parameter
                start += num_param_elements

        if self.annealing:
            self.update_annealed_param()

        if not self.stochastic:
            return None
        
        ######################################################################
        ### following part of function step() only relevant if stochastic langevin noise needs to be added
        
        
        n_particles = K.shape[0]
        n_parameters = particles.shape[1]
        jitter = 1e-9
        try:
            L_kx = torch.linalg.cholesky(K)
            # return 0, cholesky
            alpha = 0
        except Exception:
            while jitter < 1.0:
                try:
                    L_kx = torch.linalg.cholesky(K + jitter * torch.eye(K.shape[0]))
                    print('CHOLESKY: Matrix not positive-definite. Adding alpha = %.2E' % jitter)
                    # return jitter, cholesky
                    alpha = jitter
                    break
                except Exception:
                    jitter = jitter * 10

            if jitter >= 1.0:
                raise Exception('CHOLESKY: Factorization failed.')

        if alpha != 0:
            K += alpha * torch.eye(n_particles)

        Noise = torch.normal(0, 1, size=(n_parameters, n_particles))    

        v_stc = torch.sqrt(torch.tensor(2/n_particles)) * torch.einsum('mn, in -> im', L_kx, Noise).reshape(n_particles, n_parameters)
        
        for model, grads in zip(self.particle_ensemble.models, v_stc):
            flat_params = torch.cat([p.view(-1) for p in model.parameters()])
        
            start = 0  # Start index for slicing gradients from grads
            for param in model.parameters():
                # Number of elements in the current parameter
                num_param_elements = param.numel()
                end = start + num_param_elements

                # Slice the gradient for the current parameter
                grad_for_param = grads[start:end].view(param.size())
                
                # update the parameters 
                if param.grad is None:
                    continue
                else:
                    param.data.add_(torch.sqrt(self.param_groups[0]['lr']) , grad_for_param)
                    # param.data.add_(torch.sqrt(self.current_lr) , grad_for_param)
                
                # Update the start index for the next parameter
                start += num_param_elements


        return None

    def zero_grad(self):
        for model in self.particle_ensemble.models:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
    def update_annealed_param(self):
        # print('self.annealed_param:=',self.annealed_param)
        self.current_step += 1
        p = 4
        self.annealed_param = math.tanh((self.current_step/(self.annealing_period*0.2))**p)
    
    def reduce_lr(self):
        """Reduce the learning rate by half."""
        self.current_lr *= 1.
        for param_group in self.param_groups:
            param_group['lr'] = torch.tensor(float(self.current_lr), dtype=torch.float64)
        print(f"\n\n LEARNING RATE REDUCED TO:{self.current_lr} \n\n")

    def get_lr(self):
        """Get the current learning rate."""
        return self.current_lr



class SVN_laplace(optim.Optimizer):
    def __init__(self, particle_ensemble, lr=1e-3,  gamma=-1, config={}, exponential_factor=1,  stochastic = False, annealing=False, annealing_period=500, annealing_min=0.0, annealing_max=1.0):
        
        self.particle_ensemble = particle_ensemble
        
        self.stochastic = stochastic # add langevin noise to the gradients transforming into MCMC
        self.config = config
        
        # Create a single param group for all particles,  TODO: why is this?????? 
        param_group = {'params': self.particle_ensemble.parameters(), 'lr': torch.tensor(float(lr), dtype=torch.float64)}

        self.steps = 0
        self.exponential_factor = exponential_factor

        defaults = dict(lr=torch.tensor(float(lr), dtype=torch.float64), gamma=gamma)
        super(SVN_laplace, self).__init__(self.particle_ensemble.parameters(), defaults)
        
        # annealing parameters
        self.annealing = annealing # bool flag for annealing 
        self.annealing_period = annealing_period # total duration (= num_epochs*batch_size for hyperbolic tanget annealing schedule)
        self.annealing_min = annealing_min
        self.annealing_max = annealing_max
        self.current_step = 0
        self.annealed_param = self.annealing_min # parameter that is annealed

        # New attributes for learning rate schedule
        self.initial_lr = torch.tensor(float(lr))
        self.current_lr = torch.tensor(float(lr))


    
    def step(self, train_batch):
        with torch.no_grad():
            # Get flattened parameters for each particle
            particles = []
            particles_grad = []
            for model in self.particle_ensemble.models:
                particle_params = torch.cat([p.view(-1) for p in model.parameters()])
                particles.append(particle_params)
                particle_grad_params = torch.cat([p.grad.view(-1) for p in model.parameters()])
                particles_grad.append(particle_grad_params)

            particles = torch.stack(particles)  # Shape: (N, D) where N is number of particles and D is number of parameters
            particles_grad = torch.stack(particles_grad) # Shape: (N, D) gradient of params of each particles flattened


            # print('shape of particles variable:',particles.shape)
            # print('shape of particles_grad variable:',particles_grad.shape)

            # Compute pairwise distances
            # dist = torch.cdist(particles, particles)
            # print('pairwise distances shape:',dist.shape)
            
            # Convert to numpy for scikit-learn
            particles_np = particles.cpu().numpy()

            # Compute gamma if not provided
            gamma = self.defaults['gamma']
            if gamma < 0:
                gamma = 1.0 / particles_np.shape[1]

            # Compute kernel matrix using scikit-learn
            K = rbf_kernel(particles_np, gamma=gamma)

            # Convert back to PyTorch tensor
            K = torch.tensor(K, dtype=particles.dtype, device=particles.device)

            # Compute gradient of kernel
            grad_K = -K.unsqueeze(-1) * (particles.unsqueeze(1) - particles.unsqueeze(0))

            # print('K',K, 'shape:',K.shape) # NxN
            # print('grad_K',grad_K, 'shape:',grad_K.shape) # NxNxD

            if self.annealing:
                v_svgd =  self.annealed_param*(torch.einsum('mn, mo -> no', K, particles_grad)/K.shape[0]) - (torch.mean(grad_K, dim=0)) # shape = N x D ==> num_particle x num_parameters_per_model
            else:
                v_svgd =  (torch.einsum('mn, mo -> no', K, particles_grad)/K.shape[0] ) - (torch.mean(grad_K, dim=0)) # shape = N x D ==> num_particle x num_parameters_per_model

        
        hessian_tensor = self.get_laplace_hessian(train_batch=train_batch) #  diagonal approximation , N x D
        # print(hessian_tensor[0])
        # add code to compute v_svn from v_svgd, using block diagonal approx
        alphas = self.solve_block_linear_system(hessian_tensor, v_svgd, K, grad_K)
        v_svn = torch.einsum('xd, xn -> nd', alphas, K) #(n_particles, n_parameters)

        # print(v_svn[0])
        


        # update parameters below
        with torch.no_grad():
            for model, grads in zip(self.particle_ensemble.models, v_svn):
                flat_params = torch.cat([p.view(-1) for p in model.parameters()])
                
                start = 0  # Start index for slicing gradients from grads
                for param in model.parameters():
                    # Number of elements in the current parameter
                    num_param_elements = param.numel()
                    end = start + num_param_elements

                    # Slice the gradient for the current parameter
                    grad_for_param = grads[start:end].view(param.size())
                    
                    # update the parameters 
                    if param.grad is None:
                        print('grad is none LOLLL')
                        continue
                    else:
                        effective_lr = self.param_groups[0]['lr'] * (self.exponential_factor** self.steps)
                        param.data.add_( effective_lr ,grad_for_param)
                    
                    # Update the start index for the next parameter
                    start += num_param_elements

            if self.annealing:
                self.update_annealed_param()
                
            if not self.stochastic:
                return None
        
            ######################################################################
            ### following part of function step() only relevant if stochastic langevin noise needs to be added
            
            n_particles = K.shape[0]
            n_parameters = particles.shape[1]
            # jitter = 1e-9
            # try:
            #     L_kx = torch.linalg.cholesky(K)
            #     # return 0, cholesky
            #     alpha = 0
            # except Exception:
            #     while jitter < 1.0:
            #         try:
            #             L_kx = torch.linalg.cholesky(K + jitter * torch.eye(K.shape[0]))
            #             print('CHOLESKY: Matrix not positive-definite. Adding alpha = %.2E' % jitter)
            #             # return jitter, cholesky
            #             alpha = jitter
            #             break
            #         except Exception:
            #             jitter = jitter * 10

            #     if jitter >= 1.0:
            #         raise Exception('CHOLESKY: Factorization failed.')

            # if alpha != 0:
            #     K += alpha * torch.eye(n_particles)

            # Noise = torch.normal(0, 1, size=(n_parameters, n_particles))    



            # v_stc = torch.sqrt(torch.tensor(2/n_particles)) * torch.einsum('mn, in -> im', L_kx, Noise).reshape(n_particles, n_parameters)
            n_particles = K.shape[0]
            n_parameters = particles.shape[1]

            B = torch.normal(0, 1, size=(n_particles*n_particles,))
            tmp1 = self.solve_block_linear_system(hessian_tensor,v_svgd, K, grad_K).reshape(n_particles, n_parameters)
            # print(tmp1.shape)
            v_stc = torch.sqrt(torch.tensor(2/n_particles)) * torch.einsum('mn, ni -> mi', K, tmp1)

            
            for model, grads in zip(self.particle_ensemble.models, v_stc):
                flat_params = torch.cat([p.view(-1) for p in model.parameters()])
            
                start = 0  # Start index for slicing gradients from grads
                for param in model.parameters():
                    # Number of elements in the current parameter
                    num_param_elements = param.numel()
                    end = start + num_param_elements

                    # Slice the gradient for the current parameter
                    grad_for_param = grads[start:end].view(param.size())
                    
                    # update the parameters 
                    if param.grad is None:
                        continue
                    else:
                        param.data.add_(torch.sqrt(self.param_groups[0]['lr']) , grad_for_param)
                    
                    # Update the start index for the next parameter
                    start += num_param_elements


            return None

    def zero_grad(self):
        for model in self.particle_ensemble.models:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
    
    def get_laplace_hessian(self, train_batch):
        hessian_list = []

        
        # laplace_dataloader = DataLoader(data_utils.TensorDataset(train_batch[0].squeeze(1), train_batch[1]), batch_size= 2)

        for model in self.particle_ensemble.models:

            if self.config['is_classification']:
                laplace_particle_model = DiagLaplace(model, 
                                                        likelihood='classification')
                laplace_particle_model.fit(DataLoader(data_utils.TensorDataset(train_batch[0].squeeze(1), train_batch[1].squeeze(1).long()), batch_size=train_batch[0].shape[0]))
            else:
                laplace_particle_model = DiagLaplace(model, 
                                                        likelihood='regression')
                laplace_particle_model.fit(DataLoader(data_utils.TensorDataset(train_batch[0].squeeze(1), train_batch[1]), batch_size=train_batch[0].shape[0]))
            
            # print('batch size:',train_batch[0].shape[0])
            # print('train_batch[0].shape:',train_batch[0].squeeze(1).shape)
            # print('train_batch[1].shape:',train_batch[1].shape)

            

            particle_hessian = laplace_particle_model.posterior_precision
            hessian_list.append(particle_hessian)
            
        hessians_tensor = torch.cat(hessian_list, dim=0)
        hessians_tensor = hessians_tensor.reshape(self.particle_ensemble.num_particles, self.particle_ensemble.num_params)
        return hessians_tensor


    def solve_block_linear_system(self,hessian_tensor,v_svgd, K, grad_K):
        """
        hessians_tensor: NxD diagonal approxmiation for each particle
        v_svgd: NxD update for each particle
        K: NxN kernel gram matrix

        returns: 
        """


         #(n_particles, n_parameters)
        # print('hessians_tensor.shape',hessians_tensor.shape)

        # now compute v_svn by solving block linear system of equations
        # N, D = v_svgd.shape
        # alpha_list = []
        # cg_maxiter = 50     
        # for i in range(self.particle_ensemble.num_particles):
        #     v_svgd_part = v_svgd[i].squeeze().detach().cpu().flatten().numpy()
        #     squared_kernel = K**2
        #     H_op_part = scipy.sparse.linalg.LinearOperator((D, D), matvec=lambda x: diag_hessian_matvec_block(x, squared_kernel[i][i],grad_K[i],hessians_tensor[i], 'cpu'))
        #     alpha_part, _ = scipy.sparse.linalg.cg(H_op_part, v_svgd_part, maxiter=cg_maxiter)
        #     alpha_part = torch.tensor(alpha_part, dtype=torch.float32).to('cpu')
        #     alpha_list.append(alpha_part)

        # alphas = torch.stack(alpha_list, dim=0).view(self.particle_ensemble.num_particles, -1)
        # alphas_reshaped = alphas.view(self.particle_ensemble.num_particles, -1) #(n_particles, n_parameters)
        # v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K) #(n_particles, n_parameters)

        # print('v_svn:',v_svn.shape)

        # using sherman morrison formula:
        squared_kernel = K**2
        epsilon = 1e-6  # Small value for numerical stability
        alpha_list = []
        for i in range(self.particle_ensemble.num_particles):
            squared_kernel_i = squared_kernel[i][i].clone().detach().float().to("cpu")
            grad_K_i = grad_K[i][i].clone().detach().float().to("cpu")
            diag_hessian = hessian_tensor[i].clone().detach().float().to("cpu")


            D = squared_kernel_i * diag_hessian # Compute the diagonal matrix D
            D_inv = 1.0 / (D + epsilon)  # Adding epsilon directly to D
            u_T_D_inv_u = torch.sum(D_inv * grad_K_i**2)  # Computing u^T D^-1 u, which is a scalar
            D_inv_u = D_inv * grad_K_i # Compute the outer product D^-1 u u^T D^-1

            
            scaling_factor = 1 / (1 + u_T_D_inv_u + epsilon)
            term1= D_inv * v_svgd[i]
            term2 = torch.sum(D_inv_u * v_svgd[i])
            alpha_part = term1 - (scaling_factor * term2) * D_inv_u
            #outer_product = D_inv_u @ D_inv_u.t()
            #A_inv = D_inv - outer_product / (1 + u_T_D_inv_u + epsilon)  # Add epsilon to the denominator for stability
            #alpha_part = A_inv @ v_svgd[i]
            alpha_list.append(alpha_part)
        alphas = torch.stack(alpha_list, dim=0).view(self.particle_ensemble.num_particles, -1)
        # alphas_reshaped = alphas.view(self.particle_ensemble.num_particles, -1) #(n_particles, n_parameters)
        # v_svn = torch.einsum('xd, xn -> nd', alphas, K) #(n_particles, n_parameters)

        
        return alphas

    def update_annealed_param(self):
        # print('self.annealed_param:=',self.annealed_param)
        self.current_step += 1
        p = 4
        self.annealed_param = math.tanh((self.current_step/(self.annealing_period*0.2))**p)

    def reduce_lr(self):
        """Reduce the learning rate by half."""
        self.current_lr *= 1.
        for param_group in self.param_groups:
            param_group['lr'] = torch.tensor(float(self.current_lr), dtype=torch.float64)
        print(f"\n\n LEARNING RATE REDUCED TO:{self.current_lr} \n\n")

    def get_lr(self):
        """Get the current learning rate."""
        return self.current_lr

class SVN_adam(optim.Optimizer):
    def __init__(self, particle_ensemble, lr=1e-3,  gamma=-1, beta=0.8, config={},  stochastic = False):
        
        self.particle_ensemble = particle_ensemble
        
        self.stochastic = stochastic # add langevin noise to the gradients transforming into MCMC
        self.config = config

        # Create a single param group for all particles, 
        param_group = {'params': self.particle_ensemble.parameters(), 'lr': torch.tensor(float(lr), dtype=torch.float64)}

        
        self.steps = 0
        
        self.exp_avg = torch.zeros((particle_ensemble.num_particles, particle_ensemble.num_params)) # first moment

        self.exp_avg_sq = torch.zeros((particle_ensemble.num_particles, particle_ensemble.num_params)) # second moment

        self.beta = beta
        defaults = dict(lr=lr, gamma=gamma)

        super(SVN_adam, self).__init__(self.particle_ensemble.parameters(), defaults)
        

    @torch.no_grad()    
    def step(self, train_batch):
        # Get flattened parameters for each particle
        particles = []
        particles_grad = []
        for model in self.particle_ensemble.models:
            particle_params = torch.cat([p.view(-1) for p in model.parameters()])
            particles.append(particle_params)
            particle_grad_params = torch.cat([p.grad.view(-1) for p in model.parameters()])
            particles_grad.append(particle_grad_params)

        particles = torch.stack(particles)  # Shape: (N, D) where N is number of particles and D is number of parameters
        particles_grad = torch.stack(particles_grad) # Shape: (N, D) gradient of params of each particles flattened


        # print('shape of particles variable:',particles.shape)
        # print('shape of particles_grad variable:',particles_grad.shape)

        # Compute pairwise distances
        # dist = torch.cdist(particles, particles)
        # print('pairwise distances shape:',dist.shape)
        
        # Convert to numpy for scikit-learn
        particles_np = particles.cpu().numpy()

        # Compute gamma if not provided
        gamma = self.defaults['gamma']
        if gamma < 0:
            gamma = 1.0 / particles_np.shape[1]
            print('effective gamma:',gamma)

        # Compute kernel matrix using scikit-learn
        K = rbf_kernel(particles_np, gamma=gamma)

        # Convert back to PyTorch tensor
        K = torch.tensor(K, dtype=particles.dtype, device=particles.device)

        # Compute gradient of kernel
        grad_K = -K.unsqueeze(-1) * (particles.unsqueeze(1) - particles.unsqueeze(0))

        # print('K',K, 'shape:',K.shape) # NxN
        # print('grad_K',grad_K, 'shape:',grad_K.shape) # NxNxD
        
        

        v_svgd =  (torch.einsum('mn, mo -> no', K, particles_grad)/K.shape[0] ) - ( torch.mean(grad_K, dim=0)) # shape = N x D ==> num_particle x num_parameters_per_model

        # if len()
         
        
        # add code to compute v_svn from v_svgd, using block diagonal approx
        hessian_tensor = self.get_adam_hessian(particles_grad)
        # print(hessian_tensor[0])

        alphas = self.solve_block_linear_system(hessian_tensor, v_svgd, K, grad_K)

        v_svn = torch.einsum('xd, xn -> nd', alphas, K) #(n_particles, n_parameters)
        # print(v_svn[0])
        



        # update parameters below
    
        for model, grads in zip(self.particle_ensemble.models, v_svn):
            flat_params = torch.cat([p.view(-1) for p in model.parameters()])
            
            start = 0  # Start index for slicing gradients from grads
            for param in model.parameters():
                # Number of elements in the current parameter
                num_param_elements = param.numel()
                end = start + num_param_elements

                # Slice the gradient for the current parameter
                grad_for_param = grads[start:end].view(param.size())
                
                # update the parameters 
                if param.grad is None:
                    print('grad is none LOLLL')
                    continue
                else:
                    param.data.add_(self.param_groups[0]['lr'],grad_for_param)
                
                # Update the start index for the next parameter
                start += num_param_elements


        if not self.stochastic:
            # print('not adding stochastic part')
            return None
        
        ######################################################################
        ### following part of function step() only runs if self.stochastic=True langevin noise needs to be added
        
        
        n_particles = K.shape[0]
        n_parameters = particles.shape[1]
        jitter = 1e-9
        try:
            L_kx = torch.linalg.cholesky(K)
            # return 0, cholesky
            alpha = 0
        except Exception:
            while jitter < 1.0:
                try:
                    L_kx = torch.linalg.cholesky(K + jitter * torch.eye(K.shape[0]))
                    print('CHOLESKY: Matrix not positive-definite. Adding alpha = %.2E' % jitter)
                    # return jitter, cholesky
                    alpha = jitter
                    break
                except Exception:
                    jitter = jitter * 10

            if jitter >= 1.0:
                raise Exception('CHOLESKY: Factorization failed.')

        if alpha != 0:
            K += alpha * torch.eye(n_particles)

        Noise = torch.normal(0, 1, size=(n_parameters, n_particles))    

        v_stc = torch.sqrt(torch.tensor(2/n_particles)) * torch.einsum('mn, in -> im', L_kx, Noise).reshape(n_particles, n_parameters)
        
        for model, grads in zip(self.particle_ensemble.models, v_stc):
            flat_params = torch.cat([p.view(-1) for p in model.parameters()])
        
            start = 0  # Start index for slicing gradients from grads
            for param in model.parameters():
                # Number of elements in the current parameter
                num_param_elements = param.numel()
                end = start + num_param_elements

                # Slice the gradient for the current parameter
                grad_for_param = grads[start:end].view(param.size())
                
                # update the parameters 
                if param.grad is None:
                    continue
                else:
                    param.data.add_(torch.sqrt(self.param_groups[0]['lr']) , grad_for_param)
                # Update the start index for the next parameter
                start += num_param_elements


        return None

    def zero_grad(self):
        for model in self.particle_ensemble.models:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
    
    def get_adam_hessian(self, particles_grad):
        # return hessian diagonal approximation (Num of particles X Num of params)

        self.steps += 1
        # print('BEFORE',self.exp_avg_sq[0])
        exp_avg_sq = self.exp_avg_sq
        exp_avg_sq.mul_(self.beta).addcmul_(particles_grad, particles_grad, value=1 - self.beta)
        bias_correction = 1 - self.beta ** self.steps

        # exp_avg = self.exp_avg.mul_(beta1).add_(particles_grad, alpha=1 - beta1)

        # print(bias_correction)
        # print('AFTER',self.exp_avg_sq[0])

        unbiased_exp_avg_sq = exp_avg_sq / bias_correction

        # The diagonal Hessian approximation
        hessian_diag_approx = (unbiased_exp_avg_sq ) .sqrt()
        # hessian_diag_approx = unbiased_exp_avg_sq
        # print(hessian_diag_approx)
        # print('hessian_diag_approx.shape:',hessian_diag_approx.shape)
        # print(hessian_diag_approx[0])
        return hessian_diag_approx

    def solve_block_linear_system(self,hessian_tensor,v_svgd, K, grad_K):
        """
        hessians_tensor: NxD diagonal approxmiation for each particle
        v_svgd: NxD update for each particle
        K: NxN kernel gram matrix

        returns: 
        """


         #(n_particles, n_parameters)
        # print('hessians_tensor.shape',hessians_tensor.shape)

        # now compute v_svn by solving block linear system of equations
        # N, D = v_svgd.shape
        # alpha_list = []
        # cg_maxiter = 50     
        # for i in range(self.particle_ensemble.num_particles):
        #     v_svgd_part = v_svgd[i].squeeze().detach().cpu().flatten().numpy()
        #     squared_kernel = K**2
        #     H_op_part = scipy.sparse.linalg.LinearOperator((D, D), matvec=lambda x: diag_hessian_matvec_block(x, squared_kernel[i][i],grad_K[i],hessians_tensor[i], 'cpu'))
        #     alpha_part, _ = scipy.sparse.linalg.cg(H_op_part, v_svgd_part, maxiter=cg_maxiter)
        #     alpha_part = torch.tensor(alpha_part, dtype=torch.float32).to('cpu')
        #     alpha_list.append(alpha_part)

        # alphas = torch.stack(alpha_list, dim=0).view(self.particle_ensemble.num_particles, -1)
        # alphas_reshaped = alphas.view(self.particle_ensemble.num_particles, -1) #(n_particles, n_parameters)
        # v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K) #(n_particles, n_parameters)

        # print('v_svn:',v_svn.shape)

        # using sherman morrison formula:
        squared_kernel = K**2
        epsilon = 1e-6  # Small value for numerical stability
        alpha_list = []
        for i in range(self.particle_ensemble.num_particles):
            squared_kernel_i = squared_kernel[i][i].clone().detach().float().to("cpu")
            grad_K_i = grad_K[i][i].clone().detach().float().to("cpu")
            diag_hessian = hessian_tensor[i].clone().detach().float().to("cpu")


            D = squared_kernel_i * diag_hessian # Compute the diagonal matrix D
            D_inv = 1.0 / (D + epsilon)  # Adding epsilon directly to D
            u_T_D_inv_u = torch.sum(D_inv * grad_K_i**2)  # Computing u^T D^-1 u, which is a scalar
            D_inv_u = D_inv * grad_K_i # Compute the outer product D^-1 u u^T D^-1

            
            scaling_factor = 1 / (1 + u_T_D_inv_u + epsilon)
            term1= D_inv * v_svgd[i]
            term2 = torch.sum(D_inv_u * v_svgd[i])
            alpha_part = term1 - (scaling_factor * term2) * D_inv_u
            #outer_product = D_inv_u @ D_inv_u.t()
            #A_inv = D_inv - outer_product / (1 + u_T_D_inv_u + epsilon)  # Add epsilon to the denominator for stability
            #alpha_part = A_inv @ v_svgd[i]
            alpha_list.append(alpha_part)
        alphas = torch.stack(alpha_list, dim=0).view(self.particle_ensemble.num_particles, -1)
        # alphas_reshaped = alphas.view(self.particle_ensemble.num_particles, -1) #(n_particles, n_parameters)
        # v_svn = torch.einsum('xd, xn -> nd', alphas, K) #(n_particles, n_parameters)

        
        return alphas
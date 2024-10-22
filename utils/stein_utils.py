import scipy
import numpy as np
import torch
 

def diag_hessian_matvec_block(input, squared_kernel, grad_K_i, diag_hessian, device):
        
        input = torch.tensor(input).float().detach().to(device)
        squared_kernel = squared_kernel.clone().detach().float().to(device)
        grad_K_i = grad_K_i.clone().detach().float().to(device)
        diag_hessian = diag_hessian.clone().detach().float().to(device)
        kernel_grads_vector = torch.matmul(torch.matmul(grad_K_i.T, grad_K_i), input)
        kernel_weght_param_vector = squared_kernel * input
        hess_vector = diag_hessian * kernel_weght_param_vector
        update = hess_vector + kernel_grads_vector

        return update.detach().cpu().numpy()
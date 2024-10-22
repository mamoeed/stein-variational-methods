"""
Model architectures
MLP ensembles - 
Resnet ensembles
transformers?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetworkEnsemble(nn.Module):
    def __init__(self, input_size, output_size, num_particles, hidden_size=50):
        super(NeuralNetworkEnsemble, self).__init__()
        
        
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            ) for _ in range(num_particles)
        ])
        
        self.num_params = sum(p.numel() for p in self.models[0].parameters() if p.requires_grad)
        self.num_particles = num_particles
        print('initialised a neural network ensemble with', num_particles, 'particles each model having',
              self.num_params, 'parameters')
    
    def forward(self, x):
        outputs = torch.stack([model(x) for model in self.models])
        # outputs shape: (num_models, batch_size, output_dim)

        return outputs



class LeNetEnsemble(nn.Module):
    def __init__(self, output_size, num_particles):
        super(LeNetEnsemble, self).__init__()
        
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
                nn.Tanh(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(6, 16, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(16 * 5 * 5, 120),
                nn.Tanh(),
                nn.Linear(120, 84),
                nn.Tanh(),
                nn.Linear(84, output_size)
            ) for _ in range(num_particles)
        ])
        
        self.num_params = sum(p.numel() for p in self.models[0].parameters() if p.requires_grad)
        self.num_particles = num_particles
        print('Initialized a Le-Net ensemble with', num_particles, 'particles, each model having',
              self.num_params, 'parameters')
    
    def forward(self, x):
        outputs = torch.stack([model(x) for model in self.models])
        # outputs shape: (num_models, batch_size, output_dim)
        return outputs
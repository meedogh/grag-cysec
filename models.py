import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, max_resources):
        super(DQN, self).__init__()
        self.max_resources = max_resources
        self.output_dim = output_dim 

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim * max_resources) 
        )

    def forward(self, x):
        return self.net(x)  # shape: [batch_size, max_resources * n_nodes]


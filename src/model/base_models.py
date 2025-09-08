import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, n_hidden, dim_hidden, output_size):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for _ in range(n_hidden):
            layers.append(nn.Linear(prev_size, dim_hidden))
            layers.append(nn.Tanh())
            prev_size = dim_hidden
            
        # Add output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
import torch
from torch.utils.data import Dataset

class SingleSimDataset(Dataset):
    def __init__(self, data, indices):
        self.X = data['X'].flatten()[indices]
        self.Y = data['Y'].flatten()[indices]
        self.t = data['t'].flatten()[indices]
        self.V = data['V'].flatten()[indices]
        self.W = data['W'].flatten()[indices]

        # Stack into input-output pairs.
        self.inputs = torch.stack([self.X, self.Y, self.t], dim=1)  # shape: (N*T, 3)
        self.outputs = torch.stack([self.V, self.W], dim=1)          # shape: (N*T, 2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'input': self.inputs[idx],
            'output': self.outputs[idx],
            'node_count': self.inputs.shape[0]
        }
    
    def get_stats(self):
        """Returns the computed statistics if stats_flag was True."""
        stats = {
            'X': {
                'mean': self.X.mean().item(),
                'std': self.X.std().item(),
                'min': self.X.min().item(),
                'max': self.X.max().item()
            },
            'Y': {
                'mean': self.Y.mean().item(),
                'std': self.Y.std().item(),
                'min': self.Y.min().item(),
                'max': self.Y.max().item()
            },
            't': {
                'mean': self.t.mean().item(),
                'std': self.t.std().item(),
                'min': self.t.min().item(),
                'max': self.t.max().item()
            },
            'V': {
                'mean': self.V.mean().item(),
                'std': self.V.std().item(),
                'min': self.V.min().item(),
                'max': self.V.max().item()
            },
            'W': {
                'mean': self.W.mean().item(),
                'std': self.W.std().item(),
                'min': self.W.min().item(),
                'max': self.W.max().item()
            }
        }
        return stats    
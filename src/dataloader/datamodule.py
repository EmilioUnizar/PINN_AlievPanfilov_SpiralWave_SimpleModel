import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import numpy as np
from src.dataloader.dataset import SingleSimDataset
class PINNDataModule(LightningDataModule):
    def __init__(self, dataset_dir, batch_size=32, num_workers=0, ratio=1.0):
        super().__init__()
        self.pt_file = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ratio = ratio

    def setup(self, stage=None):
        data = torch.load(self.pt_file, weights_only=False)
        n = data['X'].flatten().shape[0]
        indices = np.random.permutation(n)
        n_train = int(0.8 * n * self.ratio)  # Use ratio to control the size of the training set
        n_val = int(0.2 * n * self.ratio)  # Use ratio to control the size of the validation set

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = np.arange(n)

        self.train_dataset = SingleSimDataset(data, train_idx)
        stats = self.train_dataset.get_stats()
        self.val_dataset = SingleSimDataset(data, val_idx)
        self.test_dataset = SingleSimDataset(data, test_idx)   

        return stats

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_dataset[0]['node_count'], num_workers=1, shuffle=False)
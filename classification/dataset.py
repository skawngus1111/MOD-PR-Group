import sys

from torch.utils.data import DataLoader, random_split

import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl

class DataModule(pl.LightningDataModule) :
    def __init__(self,
                 data_path = '/media/jhnam19960514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/common material/Dataset Collection',
                 data_type='MNIST',
                 batch_size=64):
        super(DataModule, self).__init__()

        self.data_path = data_path
        self.data_type = data_type
        self.batch_size = batch_size

    def setup(self, stage=None):
        transform = transforms.Compose([transforms.ToTensor()])

        if self.data_type == 'MNIST':
            train_dataset = MNIST(self.data_path, train=True, download=True, transform=transform)
            test_dataset = MNIST(self.data_path, train=False, download=True, transform=transform)
        else :
            sys.exit()

        # train/val split
        train_dataset, val_dataset = random_split(train_dataset, [55000, 5000])

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
# imports
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

# dataLoader
class MNISTDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        MNIST(root="./data", train=True, download=True)
        MNIST(root="./data", train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(root="./data", train=True, transform = transforms.ToTensor())
            self.train_dataset, self.val_dataset = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.test_dataset = MNIST(root="./data", train=False, transform = transforms.ToTensor())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
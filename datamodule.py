from torch.utils.data import DataLoader,random_split
import pytorch_lightning as pl
from torchvision import datasets
from cifardataset import CifarDataset

class CifarDataModule(pl.LightningDataModule):
    def __init__(self,data_dir,batch_size,num_workers,transform,p_uncond):
        super().__init__()
        self.data_dir=data_dir
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.transform=transform
        self.p_uncond=p_uncond

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_dir,train=True,download=True,transform=self.transform)
        datasets.CIFAR10(root=self.data_dir,train=False,download=True,transform=self.transform)

    def setup(self, stage):
        dataset=CifarDataset(data_dir=self.data_dir,
                             train=True,
                             transform=self.transform,
                             p_uncond=self.p_uncond
                )
        self.trainset,self.valset=random_split(dataset,[int(len(dataset)*0.9),len(dataset)-int(len(dataset)*0.9)])
        self.testset=CifarDataset(data_dir=self.data_dir,
                                  train=False,
                                  transform=self.transform,
                                  p_uncond=self.p_uncond
                    )
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
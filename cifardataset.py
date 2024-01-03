from torch.utils.data import Dataset,random_split
from torchvision import datasets,transforms
import numpy as np

np.random.seed(42)

class CifarDataset(Dataset):
    def __init__(self,data_dir,train=True,transform=None,p_uncond=0.1):
        super().__init__()
        self.data_dir=data_dir
        self.train=train
        self.transform=transform
        self.transform_unannotated = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.8)
        ]) 
        self.p_uncond=p_uncond
        self.data=datasets.CIFAR10(root=self.data_dir,train=self.train,transform=transform)
        self.annotated,self.un_annotated=random_split(self.data,[int(0.4*len(self.data)),len(self.data)-int(len(self.data)*0.4)])
        self.annotated=iter(self.annotated)
        self.un_annotated=iter([(self.transform_unannotated(data),10) for data,_ in iter(self.un_annotated)])

    def __len__(self):
        return len(self.data)

    #use augmentations
    def __getitem__(self, idx):
        prob=np.random.random()
        if prob < 1-self.p_uncond :
            sample=next(self.annotated)
        else:
            sample=next(self.un_annotated)
        return sample
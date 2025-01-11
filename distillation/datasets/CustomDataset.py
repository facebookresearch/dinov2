import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, img_dir: str = "/home/arda/.cache/kagglehub/datasets/ardaerendoru/gtagta/versions/1/GTA5/GTA5/images", transform: Optional[transforms.Compose] = None):

        self.img_dir = img_dir
        self.transform = transform
        
        # Get all image files
        self.images = []
        for img_name in os.listdir(self.img_dir):
            if img_name.endswith(('.jpg', '.png')):
                self.images.append(os.path.join(self.img_dir, img_name))
                
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image
    

import lightning as L
from torch.utils.data import DataLoader, random_split
from distillation.datasets.CustomDataset import CustomDataset
from distillation.datasets.collate_fn import collate_data_and_cast

class CustomDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        transform,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_split: float = 0.99
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

    def setup(self, stage=None):
        dataset = CustomDataset(img_dir=self.data_dir, transform=self.transform)
        
        train_size = int(self.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            dataset, 
            [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_data_and_cast
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_data_and_cast
        )

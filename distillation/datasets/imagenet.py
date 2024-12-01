import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional
from torchvision import transforms
from datasets import load_dataset
import datasets
import numpy as np
from torch.utils.data import Dataset, Subset
import random

class ImageNetDataset(Dataset):
    def __init__(self, type='train', transform: Optional[transforms.Compose] = None, num_samples: Optional[int] = None):
        self.dataset = load_dataset('imagenet-1k', trust_remote_code=True)
        if type == 'train':
            self.dataset = self.dataset['train']
        elif type == 'validation':
            self.dataset = self.dataset['validation']
        else:
            self.dataset = self.dataset['test']

        if num_samples is not None:
            # Randomly sample indices
            indices = random.sample(range(len(self.dataset)), num_samples)
            self.dataset = Subset(self.dataset, indices)

        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.transform:
            image = self.dataset[idx]['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return self.transform(image)
        else:
            return self.dataset[idx]['image']
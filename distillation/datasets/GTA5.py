
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional
from torchvision import transforms

class GTA5Dataset(Dataset):
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

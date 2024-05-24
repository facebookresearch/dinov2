import os
import pathlib
import random
from typing import List
from torch.utils.data import Dataset
from .decoders import ImageDataDecoder
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, path_preserved: List[str]=[], frac: float=0.1):
        self.root = root
        self.transform = transform
        self.path_preserved = path_preserved if isinstance(path_preserved, list) else list(path_preserved)
        self.frac = frac
        self.preserved_images = []
        self.images_list = self._get_image_list()

    def _get_image_list(self):
        images = []

        if isinstance(self.root, (str, pathlib.PosixPath)):
            try:
                p = self.root
                images.extend(self._retrieve_images(p, preserve=p in self.path_preserved, frac=self.frac))

            except OSError:
                print("The root given is nor a list nor a path")

        else:
            for p in self.root:
                try:
                    images.extend(self._retrieve_images(p, preserve=p in self.path_preserved, frac=self.frac))
                
                except OSError:
                    print(f"the path indicated at {p} cannot be found.")
       
        return images
    
    def _retrieve_images(self, path, is_valid=False, preserve=False, frac=1):
        images = []
        for root, _, files in os.walk(path):
            images_dir = []
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                    if is_valid:
                        try:
                            Image.open(os.path.join(root, file))
                            images_dir.append(os.path.join(root, file))
                        
                        except OSError:
                            print(f"Image at path {os.path.join(root, file)} could not be opened.")
                    else:
                        images_dir.append(os.path.join(root, file))
                    
                    if preserve:
                        random.seed(24)
                        random.shuffle(images_dir)
                        split_index = int(len(images_dir) * frac)
                        self.preserved_images.extend(images_dir[:split_index])
                        images.extend(images_dir[split_index:])

        return images
    
    def get_image_data(self, index: int):
        path = self.images_list[index]
        with open(path, 'rb') as f:
            image_data = f.read()

        return image_data

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, index: int):
        try:
            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can nor read image for sample {index}") from e

        if self.transform is not None:
            image = self.transform(image)

        return image

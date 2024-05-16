import os
import pathlib

from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images_list = self._get_image_list()

    def _get_image_list(self):
        images = []

        if isinstance(self.root, (str, pathlib.PosixPath)):
            try:
                images.extend(self._retrieve_images(self.root))

            except OSError:
                print("The root given is nor a list nor a path")

        else:
            for p in self.root:
                try:
                    images.extend(self._retrieve_images(p))
                
                except OSError:
                    print(f"the path indicated at {p} cannot be found.")
       
        return images
    
    def _retrieve_images(self, path, is_valid=False):
        images = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                    if is_valid:
                        try:
                            Image.open(os.path.join(root, file)).convert('RGB')
                            images.append(os.path.join(root, file))
                        
                        except OSError:
                            print(f"Image at path {os.path.join(root, file)} could not be opened.")
                    else:
                        images.append(os.path.join(root, file))
 
        return images

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        image_path = self.images_list[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image

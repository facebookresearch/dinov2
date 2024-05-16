import os
import pathlib

from torch.utils.data import Dataset
from .decoders import ImageDataDecoder
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
                            Image.open(os.path.join(root, file))
                            images.append(os.path.join(root, file))
                        
                        except OSError:
                            print(f"Image at path {os.path.join(root, file)} could not be opened.")
                    else:
                        images.append(os.path.join(root, file))
 
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

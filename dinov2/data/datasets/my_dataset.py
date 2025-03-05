import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from dinov2.train.rgb_to_raw import rgb_to_raw, raw_to_rgb

class ADK20Dataset(Dataset):
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shuffle: bool = False,
    ) -> None:
        """
        ADK20 Dataset for image classification.

        Args:
            root (str): Path to dataset directory.
            transforms (Callable, optional): Combined image and target transformations.
            transform (Callable, optional): Image transformations.
            target_transform (Callable, optional): Target transformations.
            shuffle (bool, optional): If True, shuffles the dataset. Defaults to False.
        """
        self.root = Path(root)
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

        # Collect image file paths
        print("root:", self.root)
        self.image_paths = sorted(self.root.rglob("*.jpg"))  # Adjust file format if needed
        if not self.image_paths:
            raise ValueError(f"No images found in dataset directory: {root}")

        if shuffle:
            import random
            random.shuffle(self.image_paths)

        self.true_len = len(self.image_paths)
        print(f"Loaded {self.true_len} images from {root}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Loads and returns an image, target, and filepath.

        Args:
            index (int): Dataset index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]: (image, target, filepath)
        """
        adjusted_index = index % self.true_len  # Avoid division by zero error
        filepath = str(self.image_paths[adjusted_index])
        # print("filepath:", filepath)
        try:
            image = Image.open(filepath).convert("RGB")
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
            return self.__getitem__((index + 1) % self.true_len)  # Skip to next valid image

        if self.transform:
            image = self.transform(image)

        target = torch.zeros((1,))  # Modify if ADK20 has labels

        if self.target_transform:
            target = self.target_transform(target)

        # raw_image = rgb_to_raw(filepath)
        # after_raw = raw_to_rgb(raw_image)
        # print("Img:", image)
        # print(type(image), type(raw_image))
        # print(type(image), type(after_raw), image.keys())
        return image, target, filepath
        # return raw_image, target, filepath
        # return image, raw_image, target, filepath

    def __len__(self) -> int:
        return self.true_len

    def rgb_to_raw(self, image_path, local_crops_number=6):
        # print("Path:", image_path)

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return {}
        
        if len(img.shape) == 3:
            img_raw = img[:, :, 1]
        else:
            img_raw = img
        
        if img_raw.dtype != np.uint16:
            img_raw = (img_raw.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
        
        # Normalize the raw image to [0, 1]
        img_raw = img_raw.astype(np.float32) / 65535.0
        

        raw_tensor = torch.from_numpy(img_raw).unsqueeze(0)  # Shape: [1, H, W]

        output = {
            "global_crops": [raw_tensor, raw_tensor],  # Two global crops
            "global_crops_teacher": [raw_tensor, raw_tensor],
            "local_crops": [raw_tensor for _ in range(local_crops_number)],
            "offsets": ()
        }
        # print("Type: ", type(rgb_to_raw))
        return output



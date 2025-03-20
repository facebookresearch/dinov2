import os
import random
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image

class ADK20Dataset(Dataset):
    def __init__(
        self,
        root: str,
        annotations_file: str = "/home/paperspace/Documents/nika_space/ADE20K/ADEChallengeData2016/sceneCategories.txt",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shuffle: bool = False,
    ) -> None:
        """
        ADK20 Dataset for image classification with labels.

        Args:
            root (str): Path to dataset directory.
            annotations_file (str): Path to the annotations file containing image IDs and labels.
            transforms (Callable, optional): Combined image and target transformations.
            transform (Callable, optional): Image transformations.
            target_transform (Callable, optional): Target transformations.
            shuffle (bool, optional): If True, shuffles the dataset. Defaults to False.
        """
        self.root = Path(root)
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform
        print("root:", self.root)
        self.image_paths = sorted(list(self.root.rglob("*.jpg")) + list(self.root.rglob("*.JPEG")))
        if not self.image_paths:
            raise ValueError(f"No images found in dataset directory: {root}")

        # Load annotations
        self.labels = {}
        self.class_to_idx = {}
        self.idx_to_class = {}
        self._load_annotations(annotations_file)
        
        # Filter image paths to only include those with annotations
        self.image_paths = [p for p in self.image_paths if self._get_image_id(p) in self.labels]
        
        if shuffle:
            import random
            random.shuffle(self.image_paths)

        self.true_len = len(self.image_paths)
        print(f"Loaded {self.true_len} images with labels from {root}")

    def _load_annotations(self, annotations_file: str) -> None:
        """
        Load annotations from the specified file.
        
        Args:
            annotations_file (str): Path to the annotations file.
        """
        try:
            with open(annotations_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_id = parts[0]
                    class_name = parts[1]
                    
                    # Add class to mapping if not already present
                    if class_name not in self.class_to_idx:
                        idx = len(self.class_to_idx)
                        self.class_to_idx[class_name] = idx
                        self.idx_to_class[idx] = class_name
                    
                    # Store label for this image
                    self.labels[image_id] = self.class_to_idx[class_name]
            
            print(f"Loaded {len(self.labels)} annotations with {len(self.class_to_idx)} unique classes")
        except Exception as e:
            print(f"Error loading annotations: {e}")
            raise

    def _get_image_id(self, filepath: Path) -> str:
        """
        Extract image ID from filepath.
        
        Args:
            filepath (Path): Path to the image file.
            
        Returns:
            str: Image ID (filename without extension).
        """
        return filepath.stem

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
        
        try:
            image = Image.open(filepath).convert("RGB")
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
            return self.__getitem__((index + 1) % self.true_len)  # Skip to next valid image

        if self.transform:
            image = self.transform(image)

        # Get label for this image
        image_id = self._get_image_id(Path(filepath))
        if image_id in self.labels:
            target = torch.tensor(self.labels[image_id])
        else:
            # Use -1 as label for images without annotations
            target = torch.tensor(-1)

        if self.target_transform:
            target = self.target_transform(target)

        return image, target, filepath

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
    
    def get_targets(self) -> np.ndarray:
        """
        Returns target labels for all dataset samples.
        
        Returns:
            np.ndarray: Array of class indices for each sample.
        """
        targets = []
        for path in self.image_paths:
            image_id = self._get_image_id(path)
            if image_id in self.labels:
                targets.append(self.labels[image_id])
            else:
                targets.append(-1)  # Use -1 for unknown labels
        
        return np.array(targets, dtype=np.int64)
    
    def get_classes(self) -> List[str]:
        """
        Returns the list of class names.
        
        Returns:
            List[str]: List of class names.
        """
        return [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
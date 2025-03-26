import os
import random
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image
import json
import rawpy
class RAWNODDataset(Dataset):
    def __init__(
        self,
        root: str,
        annotations_file: str = "/home/paperspace/Documents/nika_space/raw_nod_dataset/RAW-NOD/annotations/Nikon/raw_new_Nikon750_train.json", 
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shuffle: bool = False,
    ) -> None:
        """
        RAW-NOD Dataset for image classification with labels.

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

        # Initialize attributes
        self.image_info = {}  # Fix: Ensure image_info exists
        self.labels = {}  # Mapping of image_id → class labels
        self.class_to_idx = {}  # Mapping of class names → class IDs
        self.idx_to_class = {}  # Mapping of class IDs → class names

        # Load annotations
        self._load_annotations(annotations_file)

        # Load image paths
        self.image_paths = sorted(list(self.root.rglob("*.NEF")) + list(self.root.rglob("*.JPEG")))
        if not self.image_paths:
            raise ValueError(f"No images found in dataset directory: {root}")

        # Filter image paths based on loaded annotations
        self.image_paths = [p for p in self.image_paths if self._get_image_id(p) in self.labels]

        if shuffle:
            import random
            random.shuffle(self.image_paths)

        self.true_len = len(self.image_paths)
        print(f"Loaded {self.true_len} images with labels from {root}")



    def _load_annotations(self, annotation_file: str) -> None:
        """
        Load COCO-style annotations from a JSON file.

        Args:
            annotation_file (str): Path to the COCO annotation file.
        """
        try:
            with open(annotation_file, "r") as f:
                data = json.load(f)

            self.image_info = {}  # Ensure this is initialized
            self.labels = {}  # Reset labels
            self.class_to_idx = {}
            self.idx_to_class = {}

            # Process category labels
            for category in data["categories"]:
                self.class_to_idx[category["name"]] = category["id"]
                self.idx_to_class[category["id"]] = category["name"]

            # Process images
            for img in data["images"]:
                self.image_info[img["id"]] = img  # Store image metadata
                self.labels[img["id"]] = []  # Initialize empty label list

            # Process annotations
            for anno in data["annotations"]:
                img_id = anno["image_id"]
                category_id = anno["category_id"]

                if img_id in self.labels:
                    self.labels[img_id].append(category_id)

            print(f"Loaded {len(self.labels)} annotations for {len(self.image_info)} images.")

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
            # image = Image.open(filepath).convert("RGB")
            raw = rawpy.imread(filepath)
            rgb = raw.postprocess()
            image = Image.fromarray(rgb)
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
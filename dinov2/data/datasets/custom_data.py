from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from torchvision.datasets import DatasetFolder

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")


def pil_loader(p):
    return Image.open(p).convert("RGB")


class CustomData(ExtendedVisionDataset):

    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.data = DatasetFolder(root, loader=pil_loader, extensions=["jpg"])

    def get_image_data(self, index: int) -> bytes:
        return self.data[index][0]

    def get_target(self, index: int) -> Optional[int]:
        return 0

    def __len__(self) -> int:
        return len(self.data)
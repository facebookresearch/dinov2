import torch
import numpy as np
from .extended import ExtendedVisionDataset
from typing import Any, Callable, List, Optional, Set, Tuple


class MockDataset(ExtendedVisionDataset):
    def __init__(
            self,
            *,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__("", transforms, transform, target_transform)
        self._extra_root = ""

    def get_image_data(self, index: int) -> bytes:
        return torch.rand((3,448,448)) * 255
        # return np.zeros((3,448,448))

    def get_target(self, index: int) -> Any:
        return 0

    def __len__(self):
        return 2048

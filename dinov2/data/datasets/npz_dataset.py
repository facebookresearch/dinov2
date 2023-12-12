import glob
import os
from enum import Enum
from typing import Union

import numpy as np

from dinov2.data.datasets import ImageNet
from dinov2.data.datasets.image_net import _Split


class _SplitNPZDataset(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split


class NPZDataset(ImageNet):
    Split = Union[_SplitNPZDataset]

    @property
    def _entries_path(self) -> str:
        return f"-{self._split.value.upper()}.npz"

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self.root, self._extra_root + extra_path)

    def _get_class_ids(self) -> np.ndarray:
        if self._split == _Split.TEST:
            assert False, "Class IDs are not available in TEST split"
        if self._class_ids is None:
            self._load_extra(self._class_names_path)
            self._class_names = self._data["class_id"]
        assert self._class_ids is not None
        return self._class_ids

    def _get_class_names(self) -> np.ndarray:
        if self._split == _Split.TEST:
            assert False, "Class names are not available in TEST split"
        if self._class_names is None:
            self._load_extra(self._class_names_path)
            self._class_names = self._data["class_str"]
        assert self._class_names is not None
        return self._class_names

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._load_extra(self._entries_path)
            self._entries = self._data["images"]
        assert self._entries is not None
        return self._entries

    def _load_extra(self, extra_path: str):
        extra_full_path = self._get_extra_full_path(extra_path)
        file_list = glob.glob(extra_full_path)

        concatenated_data = {}

        for file in file_list:
            data = np.load(file, mmap_mode="r")
            for key in data:
                if key in concatenated_data:
                    concatenated_data[key] = np.concatenate((concatenated_data[key], data[key]))
                else:
                    concatenated_data[key] = data[key]

        self._data = concatenated_data

    def __len__(self) -> int:
        entries = self._get_entries()
        return len(entries)

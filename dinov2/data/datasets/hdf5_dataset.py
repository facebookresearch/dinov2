import glob
import json
import os
from enum import Enum
from typing import Union, Optional, Tuple

import h5py
import numpy as np

from dinov2.data.datasets import ImageNet

_TargetHDF5Dataset = int

class _SplitHDF5Dataset(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split


class HDF5Dataset(ImageNet):
    Target = Union[_TargetHDF5Dataset]
    Split = Union[_SplitHDF5Dataset]
    hdf5_handles = {}

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        entry = entries[index]
        image_relpath = entry["path"]
        hdf5_path = entry["hdf5_file"]
        hdf5_file = self.hdf5_handles[hdf5_path]
        image_data = hdf5_file[image_relpath][()]
        return image_data

    def get_target(self, index: int) -> Optional[Target]:
        entries = self._get_entries()
        class_index = entries[index]["class_id"]
        return None if self.split == _SplitHDF5Dataset.TEST else int(class_index)

    def get_class_names(self) -> np.ndarray:
        self._get_entries()
        return self._class_names

    def get_class_ids(self) -> np.ndarray:
        self._get_entries()
        return self._class_ids

    @property
    def _entries_path(self) -> str:
        return f"-{self._split.value.upper()}.hdf5"

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self.root, self._extra_root + extra_path)

    def _get_entries(self) -> list:
        if self._entries is None:
            self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def _load_extra(self, extra_path: str):
        extra_full_path = self._get_extra_full_path(extra_path)
        file_list = glob.glob(extra_full_path)

        accumulated = []
        class_ids = []
        class_names = []

        for hdf5_file in file_list:
            file = h5py.File(hdf5_file, 'r')
            self.hdf5_handles[hdf5_file] = file
            # Read the JSON string from the 'file_index' dataset
            file_index_json = file['file_index'][()]
            file_index = json.loads(file_index_json)

            # Add the HDF5 file name to each entry and accumulate the file entries
            for entry in file_index['files']:
                entry['hdf5_file'] = hdf5_file  # Add the HDF5 file name to the entry
                accumulated.append(entry)
                class_id = entry['class_id']
                class_str = entry['class_str']
                if class_id not in class_ids:
                    class_ids.append(class_id)
                    class_names.append(class_str)

        self._entries = accumulated
        self._class_ids = class_ids
        self._class_names = class_names

    def __len__(self) -> int:
        entries = self._get_entries()
        return len(entries)

    def close(self):
        for handle in self.hdf5_handles.values():
            handle.close()

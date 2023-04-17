# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset


_Labels = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 1_281_167,
            _Split.VAL: 50_000,
            _Split.TEST: 100_000,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(self, actual_index: int, class_id: Optional[str] = None) -> str:
        dirname = self.get_dirname(class_id)
        if self == _Split.TRAIN:
            basename = f"{class_id}_{actual_index}"
        else:  # self in (_Split.VAL, _Split.TEST):
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        class_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index


class ImageNet(ExtendedVisionDataset):
    Labels = Union[_Labels]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "ImageNet.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra

        self._split = split

        entries_path = self._get_entries_path(split, root)
        self._entries = self._load_extra(entries_path)

        self._class_ids = None
        self._class_names = None

        if split == _Split.TEST:
            return

        class_ids_path = self._get_class_ids_path(split, root)
        self._class_ids = self._load_extra(class_ids_path)

        class_names_path = self._get_class_names_path(split, root)
        self._class_names = self._load_extra(class_names_path)

    @property
    def split(self) -> "ImageNet.Split":
        return self._split

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_root = self._extra_root
        extra_full_path = os.path.join(extra_root, extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_root = self._extra_root
        extra_full_path = os.path.join(extra_root, extra_path)
        os.makedirs(extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    def _get_entries_path(self, split: "ImageNet.Split", root: Optional[str] = None) -> str:
        return f"entries-{split.value.upper()}.npy"

    def _get_class_ids_path(self, split: "ImageNet.Split", root: Optional[str] = None) -> str:
        return f"class-ids-{split.value.upper()}.npy"

    def _get_class_names_path(self, split: "ImageNet.Split", root: Optional[str] = None) -> str:
        return f"class-names-{split.value.upper()}.npy"

    def find_class_id(self, class_index: int) -> str:
        assert self._class_ids is not None
        return str(self._class_ids[class_index])

    def find_class_name(self, class_index: int) -> str:
        assert self._class_names is not None
        return str(self._class_names[class_index])

    def get_image_data(self, index: int) -> bytes:
        actual_index = self._entries[index]["actual_index"]
        class_id = self.get_class_id(index)
        image_relpath = self.split.get_image_relpath(actual_index, class_id)
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Optional[_Labels]:
        class_index = self._entries[index]["class_index"]
        return None if self.split == _Split.TEST else int(class_index)

    def get_targets(self) -> Optional[np.ndarray]:
        return None if self.split == _Split.TEST else self._entries["class_index"]

    def get_class_id(self, index: int) -> Optional[str]:
        class_id = self._entries[index]["class_id"]
        return None if self.split == _Split.TEST else str(class_id)

    def get_class_name(self, index: int) -> Optional[str]:
        class_name = self._entries[index]["class_name"]
        return None if self.split == _Split.TEST else str(class_name)

    def __len__(self) -> int:
        assert len(self._entries) == self.split.length
        return len(self._entries)

    def _load_labels(self, root: str) -> List[Tuple[str, str]]:
        path = os.path.join(root, "labels.txt")
        labels = []

        try:
            with open(path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    class_id, class_name = row
                    labels.append((class_id, class_name))
        except OSError as e:
            raise RuntimeError(f'can not read labels file "{path}"') from e

        return labels

    def _dump_entries(self, split: "ImageNet.Split", root: Optional[str] = None) -> None:
        # NOTE: Using torchvision ImageFolder for consistency
        from torchvision.datasets import ImageFolder

        root = self.root
        labels = self._load_labels(root)

        if split == ImageNet.Split.TEST:
            dataset = None
            sample_count = split.length
            max_class_id_length, max_class_name_length = 0, 0
        else:
            dataset_root = os.path.join(root, split.get_dirname())
            dataset = ImageFolder(dataset_root)
            sample_count = len(dataset)
            max_class_id_length, max_class_name_length = -1, -1
            for sample in dataset.samples:
                _, class_index = sample
                class_id, class_name = labels[class_index]
                max_class_id_length = max(len(class_id), max_class_id_length)
                max_class_name_length = max(len(class_name), max_class_name_length)

        dtype = np.dtype(
            [
                ("actual_index", "<u4"),
                ("class_index", "<u4"),
                ("class_id", f"U{max_class_id_length}"),
                ("class_name", f"U{max_class_name_length}"),
            ]
        )
        entries_array = np.empty(sample_count, dtype=dtype)

        if split == ImageNet.Split.TEST:
            for index in range(sample_count):
                entries_array[index] = (index + 1, np.uint32(-1), "", "")
        else:
            class_names = {class_id: class_name for class_id, class_name in labels}

            assert dataset
            for index, _ in enumerate(dataset):
                image_full_path, class_index = dataset.samples[index]
                image_relpath = os.path.relpath(image_full_path, root)
                class_id, actual_index = split.parse_image_relpath(image_relpath)
                class_name = class_names[class_id]
                entries_array[index] = (actual_index, class_index, class_id, class_name)

        entries_path = self._get_entries_path(split, root)
        self._save_extra(entries_array, entries_path)

    def _dump_class_ids_and_names(self, split: "ImageNet.Split", root: Optional[str] = None) -> None:
        if split == ImageNet.Split.TEST:
            return

        root = self.get_root(root)
        entries_path = self._get_entries_path(split, root)
        entries_array = self._load_extra(entries_path)

        max_class_id_length, max_class_name_length, max_class_index = -1, -1, -1
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            max_class_index = max(int(class_index), max_class_index)
            max_class_id_length = max(len(str(class_id)), max_class_id_length)
            max_class_name_length = max(len(str(class_name)), max_class_name_length)

        class_count = max_class_index + 1
        class_ids_array = np.empty(class_count, dtype=f"U{max_class_id_length}")
        class_names_array = np.empty(class_count, dtype=f"U{max_class_name_length}")
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            class_ids_array[class_index] = class_id
            class_names_array[class_index] = class_name

        class_ids_path = self._get_class_ids_path(split, root)
        self._save_extra(class_ids_array, class_ids_path)

        class_names_path = self._get_class_names_path(split, root)
        self._save_extra(class_names_array, class_names_path)

    def dump_extra(self, split: "ImageNet.Split", root: Optional[str] = None) -> None:
        self._dump_entries(split, root)
        self._dump_class_ids_and_names(split, root)

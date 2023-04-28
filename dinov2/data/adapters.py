# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

from torch.utils.data import Dataset

#Defining a class 'DatasetWithEnumeratedTargets' which inherits the parent class 'Dataset'
class DatasetWithEnumeratedTargets(Dataset):
    # Constructor to initialize the object of the class
    def __init__(self, dataset):
        self._dataset = dataset

    # Method to get the image data from the dataset
    def get_image_data(self, index: int) -> bytes:
        return self._dataset.get_image_data(index)

    # Method to get the target label from the dataset, where target is a tuple of any data type and an integer value
    def get_target(self, index: int) -> Tuple[Any, int]:
        target = self._dataset.get_target(index)
        return (index, target)

    # Method to get an item (image and target) from the dataset
    def __getitem__(self, index: int) -> Tuple[Any, Tuple[Any, int]]:
        image, target = self._dataset[index]
        # If the target is None, return index as the target label
        target = index if target is None else target
        return image, (index, target)

    # Method to get the length of the dataset
    def __len__(self) -> int:
        return len(self._dataset)

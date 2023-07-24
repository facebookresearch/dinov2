# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Tuple

from torch.utils.data import Dataset


class DatasetWithEnumeratedTargets(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def get_image_data(self, index: int) -> bytes:
        return self._dataset.get_image_data(index)

    def get_target(self, index: int) -> Tuple[Any, int]:
        target = self._dataset.get_target(index)
        return (index, target)

    def __getitem__(self, index: int) -> Tuple[Any, Tuple[Any, int]]:
        image, target = self._dataset[index]
        target = index if target is None else target
        return image, (index, target)

    def __len__(self) -> int:
        return len(self._dataset)

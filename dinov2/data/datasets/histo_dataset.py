from pathlib import Path
from typing import Optional, Callable, Tuple, Any

import h5py
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from tqdm import tqdm


class HistoPatchDataset(VisionDataset):
    """
    Dataset for training "Foundation Model" for histopathology. Loads patches from h5 files.
    Assumes that each h5 file contains a dataset with patches and the number of patches is the same for all files
    (except for the last file, which can have fewer patches). Patches are cached and loaded in the order
    they are stored in the file for efficiency.
    Patches in the h5 files are assumed to be shuffled, so no shuffling is needed.
    See also `FmDistributedSampler` for distributed training.

    Args:
        root: path to the directory with h5 files
        split: dataset phase (train or val)
        internal_patch_count: expected number of patches in each h5 file
        internal_dataset: name of the internal dataset in the h5 file
        cache_size: number of files to cache in memory
        train_ratio: ratio of training files, by default 1.0 (use all files)
        file_filter: filter for file names
        seed: random seed for shuffling the file list
    """

    def __init__(self,
                 root: str,
                 split: str = "train",
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 internal_patch_count: int = 2048,
                 internal_dataset="patches",
                 cache_size: int = 16,
                 train_ratio: float = 1.0,
                 file_filter: str = None,
                 seed: int = 47):
        super().__init__(root, transforms, transform, target_transform)
        self.root_dir = Path(root)
        assert split in ["train", "val"]
        self.internal_patch_count = internal_patch_count
        self.internal_dataset = internal_dataset
        self.cache_size = cache_size

        # load file list
        self.file_list = list(self.root_dir.glob("*.h5"))
        if file_filter is not None:
            self.file_list = [f for f in self.file_list if file_filter in f.name]

        # randomly shuffle file list using a given seed
        rs = np.random.RandomState(seed)
        rs.shuffle(self.file_list)
        # split file list into train and val subsets
        train_count = int(train_ratio * len(self.file_list))
        if split == "train":
            self.file_list = self.file_list[:train_count]
        else:
            self.file_list = self.file_list[train_count:]

        # compute number of patches in the dataset
        self.num_patches = 0
        self.file_patch_count = {}
        small_patch_file_count = 0
        for file_path in tqdm(self.file_list):
            with h5py.File(file_path, "r") as f:
                patch_count = f[self.internal_dataset].shape[0]
                if patch_count != self.internal_patch_count:
                    small_patch_file_count += 1
                    assert small_patch_file_count <= 1, f"Found multiple files with patch count smaller than {self.internal_patch_count}. Only one such file is allowed!"
                self.num_patches += patch_count
                self.file_patch_count[file_path] = patch_count

        # sort files by number of patches in descending order
        self.file_list = sorted(self.file_list, key=lambda f: self.file_patch_count[f], reverse=True)

        # initialize file cache dictionary
        self.file_cache = {}

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        # find file containing the patch
        file_idx = idx // self.internal_patch_count
        patch_idx = idx % self.internal_patch_count

        file_path = self.file_list[file_idx]
        if file_path in self.file_cache:
            patches = self.file_cache[file_path]
        else:
            if len(self.file_cache) >= self.cache_size:
                # remove the first element from the cache
                k = next(iter(self.file_cache))
                self.file_cache.pop(k)
            # load patches from the file
            with h5py.File(file_path, "r") as f:
                patches = f[self.internal_dataset][:]
                self.file_cache[file_path] = patches
        img = patches[patch_idx]
        img = np.transpose(img, (1, 2, 0))
        image = Image.fromarray(img).convert(mode="RGB")
        # apply augmentations
        image, target = self.transforms(image, None)
        return image, target

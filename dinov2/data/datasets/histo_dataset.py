import io
from pathlib import Path
from typing import Optional, Callable, Tuple, Any

import boto3
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
        extra: additional parameters
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
                 extra: str = None,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 internal_patch_count: int = 2048,
                 internal_dataset="patches",
                 cache_size: int = 16,
                 train_ratio: float = 1.0,
                 file_filter: str = None):
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        assert split in ["train", "val"]
        # parse extra parameters
        if extra is not None:
            parts = extra.split(",")
            data_location = parts[0]
            file_filter = parts[1] if len(parts) > 1 else None
            assert data_location in ["local", "s3"]
        else:
            data_location = "local"

        predicate = lambda x: str(x).endswith(".h5")
        if file_filter is not None:
            predicate = lambda x: str(x).endswith(".h5") and file_filter in str(x)

        if data_location == "local":
            # load local files names
            self.file_list = list(filter(predicate, Path(root).glob("*.h5")))
            self.s3_storage = False
        else:
            self.s3_storage = True
            # load s3 files names
            s3 = boto3.resource("s3")
            bucket = s3.Bucket(root)
            self.file_list = list(filter(predicate, [obj.key for obj in bucket.objects.all()]))
            # init s3 client
            self.s3_client = boto3.client('s3')

        assert len(self.file_list) > 0, f"No files found in {root}"

        self.internal_patch_count = internal_patch_count
        self.internal_dataset = internal_dataset
        self.cache_size = cache_size

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
            patch_count = self._load_patches(file_path, return_count=True)
            if patch_count != self.internal_patch_count:
                small_patch_file_count += 1
                assert small_patch_file_count <= 2, f"Multiple files with patch count smaller than {self.internal_patch_count} are not allowed!"
            self.num_patches += patch_count
            self.file_patch_count[file_path] = patch_count

        # sort files by number of patches in descending order
        self.file_list = sorted(self.file_list, key=lambda f: self.file_patch_count[f], reverse=True)

        # initialize file cache dictionary
        self.file_cache = {}

    def _load_patches(self, file_path: str, return_count: bool = False) -> np.ndarray:
        if self.s3_storage:
            file = io.BytesIO()
            self.s3_client.download_fileobj(self.root, file_path, file)
            file.seek(0)
        else:
            file = file_path

        with h5py.File(file, "r") as f:
            if return_count:
                return f[self.internal_dataset].shape[0]
            return f[self.internal_dataset][:]

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        # find file containing the patch
        file_idx = idx // self.internal_patch_count
        patch_idx = idx % self.internal_patch_count

        file_path = self.file_list[file_idx]
        if file_path in self.file_cache:
            # cache hit
            patches = self.file_cache[file_path]
        else:
            # cache miss: load patches from the file
            if len(self.file_cache) >= self.cache_size:
                # remove the first element from the cache
                k = next(iter(self.file_cache))
                self.file_cache.pop(k)
            # load patches from the file
            patches = self._load_patches(file_path)
            self.file_cache[file_path] = patches

        # convert patch to image
        img = patches[patch_idx]
        img = np.transpose(img, (1, 2, 0))
        image = Image.fromarray(img).convert(mode="RGB")
        # apply augmentations
        image, target = self.transforms(image, None)
        return image, target

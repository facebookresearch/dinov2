from typing import TypeVar, Optional, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from dinov2.data.datasets.histo_dataset import HistoPatchDataset

T_co = TypeVar('T_co', covariant=True)


class HistoInfiniteDistributedSampler(Sampler[T_co]):
    """Foundation Model Distributed Sampler. Splits the training files equally between different ranks
    and assigns patches from the files to the ranks. Makes sure that the cache from the dataset
    is efficiently used, i.e. consecutive patches are loaded from the same file before moving to the next file.

    Args:
        dataset: Dataset used for sampling.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    """

    def __init__(self, dataset: HistoPatchDataset, shuffle: bool = True, seed: int = 0) -> None:
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        num_replicas = dist.get_world_size()
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        # create file_ids
        num_files = len(self.dataset.file_list)
        if num_files % self.num_replicas != 0:
            # make sure to split files evenly
            num_files += num_files // self.num_replicas + self.num_replicas
        # make sure file indices are circular
        self.file_ids = [i % len(self.dataset.file_list) for i in range(num_files)]
        # compute number of samples
        self.num_samples = len(self.file_ids) // self.num_replicas * self.dataset.internal_patch_count
        # create mapping from file_ids to patch indices
        self.file_patch_indices = {}
        for file_id in self.file_ids:
            start_idx = file_id * self.dataset.internal_patch_count
            end_idx = start_idx + self.dataset.internal_patch_count
            indices = list(range(start_idx, end_idx))
            self.file_patch_indices[file_id] = indices

    def __iter__(self) -> Iterator[T_co]:
        yield from self._iterator()

    def _iterator(self):
        file_ids = self.file_ids.copy()
        if self.shuffle:
            # deterministically shuffle based on the seed
            rs = np.random.RandomState(self.seed)
            rs.shuffle(file_ids)

        # split files between different ranks
        file_id_partitions = np.array_split(file_ids, self.num_replicas)
        # get file indices for the current rank
        file_ids = file_id_partitions[self.rank].tolist()
        # convert file indices to patch indices
        indices = []
        for file_id in file_ids:
            indices.extend(self.file_patch_indices[file_id])

        # shuffle indices if necessary
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(self.num_samples, generator=g)
        else:
            perm = torch.arange(self.num_samples)
        i = 0
        while True:
            yield indices[perm[i].item()]
            i += 1
            # reset the index and reshuffle if necessary
            if i >= self.num_samples:
                i = 0
                if self.shuffle:
                    # reshuffle the indices
                    perm = torch.randperm(self.num_samples, generator=g)

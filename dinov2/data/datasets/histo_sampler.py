import itertools
from typing import TypeVar, Iterator

import numpy as np
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
            num_files = (num_files // self.num_replicas + 1) * self.num_replicas
        # make sure file indices are circular
        self.file_ids = [i % len(self.dataset.file_list) for i in range(num_files)]
        if self.shuffle:
            # randomize file_ids
            rs = np.random.RandomState(seed)
            rs.shuffle(self.file_ids)
            # init random generator for shuffling indices
            self.rs = np.random.RandomState(seed + self.rank)
        # partition the files between different ranks
        file_ids = np.array_split(self.file_ids, self.num_replicas)[self.rank]
        # convert file ids to 2D patch indices array
        self.indices = []
        for file_id in file_ids:
            start_idx = file_id * self.dataset.internal_patch_count
            end_idx = start_idx + self.dataset.internal_patch_count
            self.indices.append(list(range(start_idx, end_idx)))

        # compute number of patches
        self.num_samples = len(file_ids) * self.dataset.internal_patch_count

    def __iter__(self) -> Iterator[T_co]:
        yield from self._iterator()

    def _iterator(self):
        if self.shuffle:
            self._shuffle_indices()
        index_iterator = itertools.chain(*self.indices)
        i = 0
        while True:
            yield from index_iterator
            i += 1
            # reset the index and reshuffle if necessary
            if i >= self.num_samples:
                i = 0
                if self.shuffle:
                    # reshuffle the indices
                    self._shuffle_indices()

    def _shuffle_indices(self):
        """We must shuffle within the individual rows to ensure data locality and efficient cache usage."""
        # shuffle the rows
        self.rs.shuffle(self.indices)
        # shuffle withing the individual rows
        for ind in self.indices:
            self.rs.shuffle(ind)

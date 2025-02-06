# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import itertools
from typing import Any, Optional, List, Union, Iterator
import warnings

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

import dinov2.distributed as distributed


class EpochSampler(Sampler):
    def __init__(
        self,
        *,
        size: int,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
    ):
        self._size = size
        self._sample_count = sample_count
        self._shuffle = shuffle
        self._seed = seed
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._epoch = 0

    def __iter__(self):
        count = (self._size + self._sample_count - 1) // self._sample_count
        tiled_indices = np.tile(np.arange(self._sample_count), count)
        if self._shuffle:
            seed = self._seed * self._epoch if self._seed != 0 else self._epoch
            rng = np.random.default_rng(seed)
            iterable = rng.choice(tiled_indices, self._size, replace=False)
        else:
            iterable = tiled_indices[: self._size]

        yield from itertools.islice(iterable, self._start, None, self._step)

    def __len__(self):
        return (self._size - self._start + self._step - 1) // self._step

    def set_epoch(self, epoch):
        self._epoch = epoch


def _get_numpy_dtype(size: int) -> Any:
    return np.int32 if size <= 2**31 else np.int64


def _get_torch_dtype(size: int) -> Any:
    return torch.int32 if size <= 2**31 else torch.int64


def _generate_randperm_indices(*, size: int, generator: torch.Generator):
    """Generate the indices of a random permutation."""
    dtype = _get_torch_dtype(size)
    # This is actually matching PyTorch's CPU implementation, see: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorFactories.cpp#L900-L921
    perm = torch.arange(size, dtype=dtype)
    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()

        # Always swap even if no-op
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value
        yield value


class InfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._advance = advance

    def __iter__(self):
        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator().manual_seed(self._seed)

        while True:
            iterable = _generate_randperm_indices(size=self._sample_count, generator=generator)
            yield from itertools.islice(iterable, self._start, None, self._step)


# The following function is somewhat equivalent to _new_shuffle_tensor_slice below,
# but avoids a full in-place random permutation generation.
def _shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    stop = len(tensor)
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}")

    dtype = _get_numpy_dtype(stop)
    result = np.empty(count, dtype=dtype)

    for i in range(count):
        j = torch.randint(0, i + 1, size=(1,), generator=generator).item() if i > 0 else 0

        result[i] = result[j]
        result[j] = tensor[start + i * step].item()

    return result


def _new_shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    stop = len(tensor)
    count = stop // step
    dtype = torch.int64  # Needed for using randperm result as indices
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}")
    indices = torch.randperm(count, dtype=dtype, generator=generator)
    return tensor[start::step][indices].numpy()


def _make_seed(seed: int, start: int, iter_count: int) -> int:
    # NOTE: Tried a few variants (including iter_count << 32), this one worked best.
    return seed + start + (iter_count << 24)


class ShardedInfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
        use_new_shuffle_tensor_slice: bool = False,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._advance = advance
        self._iter_count = 0
        self._shuffle_tensor_slice_fn = (
            _new_shuffle_tensor_slice if use_new_shuffle_tensor_slice else _shuffle_tensor_slice
        )

    def __iter__(self):
        iter_count = self._advance // self._sample_count
        if iter_count > 0:
            self._advance -= iter_count * self._sample_count
            self._iter_count += iter_count

        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to be keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator()

        # Always shuffle everything first
        generator.manual_seed(self._seed)
        dtype = _get_torch_dtype(self._sample_count)
        perm = torch.randperm(self._sample_count, dtype=dtype, generator=generator)

        while True:
            # Re-seed on each iteration to allow skipping whole permutations
            seed = _make_seed(self._seed, self._start, self._iter_count)
            generator.manual_seed(seed)

            iterable = self._shuffle_tensor_slice_fn(
                tensor=perm, start=self._start, step=self._step, generator=generator
            )
            yield from iterable
            self._iter_count += 1


class ShardedInfiniteBalancedSampler(Sampler):
    def __init__(
        self,
        labels: List[int],
        mode: Union[str, int] = "downsampling",
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
        use_new_shuffle_tensor_slice: bool = False,
    ):
        """
        A sharded infinite sampler that can optionally perform balanced (stratified) class sampling.

        Args:
            labels: List of class labels for the dataset.
            mode: "downsampling", "upsampling", or an integer specifying the number of samples per class per cycle.
                  - "downsampling": each class is sampled using the count equal to the minimum available samples.
                  - "upsampling": each class is sampled using the count equal to the maximum available samples.
            shuffle: Whether to shuffle the balanced samples each cycle.
            seed: Random seed for shuffling.
            start: Shard start index (defaults to global rank).
            step: Shard step (defaults to global size).
            advance: Number of initial samples to skip.
            use_new_shuffle_tensor_slice: Switch to the alternate shuffle slice function.
        """
        super().__init__(labels)
        self._labels = np.array(labels)
        self._seed = seed
        self._shuffle = shuffle

        # Sharding info.
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._advance = advance

        # Choose the slice function.
        self._shuffle_tensor_slice_fn = (
            _new_shuffle_tensor_slice if use_new_shuffle_tensor_slice else _shuffle_tensor_slice
        )

        # Cycle count.
        self._iter_count = 0

        # Balanced sampling configuration.
        self._unique_labels = np.unique(self._labels)
        self._lbl2idx = {lbl: np.where(self._labels == lbl)[0] for lbl in self._unique_labels}
        self._sorted_labels = sorted(self._unique_labels)

        # Determine samples per class.
        if isinstance(mode, str):
            if mode == "downsampling":
                self._samples_per_class = min(len(idxs) for idxs in self._lbl2idx.values())
            elif mode == "upsampling":
                self._samples_per_class = max(len(idxs) for idxs in self._lbl2idx.values())
            else:
                raise ValueError(f"mode='{mode}' must be 'downsampling', 'upsampling', or an integer.")
        elif isinstance(mode, int):
            self._samples_per_class = mode
        else:
            raise ValueError(f"mode must be str or int, got {type(mode)}.")

        # The size of one balanced "cycle" (nominal epoch length).
        self._sample_count = self._samples_per_class * len(self._sorted_labels)

    def __iter__(self) -> Iterator[int]:
        """
        Main entry point for iteration.
        - Possibly skips entire cycles if 'advance' is large.
        - Then yields from either _iterator() or _shuffled_iterator(),
          slicing off the first 'advance' items.
        """
        # Skip entire cycles if _advance is large.
        iter_count = self._advance // self._sample_count
        if iter_count > 0:
            self._advance -= iter_count * self._sample_count
            self._iter_count += iter_count

        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self) -> Iterator[int]:
        """
        Non-shuffling infinite iterator.
        Builds a balanced set each cycle (down/up sample) without final shuffling, then shards it.
        """
        rng = np.random.default_rng(self._seed)
        while True:
            indices = []
            for lbl in self._sorted_labels:
                idxs = self._lbl2idx[lbl]
                replace = (self._samples_per_class > len(idxs))
                chosen = rng.choice(idxs, self._samples_per_class, replace=replace)
                indices.extend(chosen)
            # Shard for this rank.
            for idx in indices[self._start::self._step]:
                yield int(idx)
            self._iter_count += 1

    def _shuffled_iterator(self) -> Iterator[int]:
        """
        Shuffling infinite iterator.
        Each cycle picks new balanced subsets from each class, shuffles them,
        and shards them.
        """
        # Torch generator for slicing.
        generator = torch.Generator()
        while True:
            seed = _make_seed(self._seed, self._start, self._iter_count)
            generator.manual_seed(seed)
            # Create a single NumPy RNG for the entire cycle.
            rng = np.random.RandomState(seed)
            indices_np = []
            for lbl in self._sorted_labels:
                idxs = self._lbl2idx[lbl]
                replace = (self._samples_per_class > len(idxs))
                chosen = rng.choice(idxs, self._samples_per_class, replace=replace)
                indices_np.extend(chosen)
            # Convert to torch tensor.
            cycle_size = len(indices_np)
            dtype = _get_torch_dtype(cycle_size)
            indices_tensor = torch.tensor(indices_np, dtype=dtype)
            # Final shuffle and shard using the slice function.
            iterable = self._shuffle_tensor_slice_fn(
                tensor=indices_tensor,
                start=self._start,
                step=self._step,
                generator=generator
            )
            yield from iterable
            self._iter_count += 1

    def __len__(self) -> int:
        """
        Returns the nominal size of one 'cycle': samples_per_class * number of classes.
        """
        return self._sample_count

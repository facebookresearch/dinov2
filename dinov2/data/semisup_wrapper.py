import math
from typing import Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist


class SemiSupervisedWrapper(torch.utils.data.Dataset):
    """
    Wrapper for datasets that adds semi-supervised functionality.

    This wrapper creates a supervised/unsupervised split of the dataset and adds
    an 'is_supervised' column to the underlying dataset to track which samples
    are labeled.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        main_label_col: str,
        supervised_proportion: float = 0.1,
        seed: int = 813846,
        labels_per_class: Optional[int] = None,
    ):
        """
        Initialize the semi-supervised wrapper.

        Args:
            dataset: The underlying dataset to wrap
            main_label_col: Name of the main label column in the dataset
            supervised_proportion: Proportion of data to use as supervised (0-1)
            seed: Random seed for reproducibility
            labels_per_class: If set, use this many labels per class instead of proportion
        """
        assert (
            0 <= supervised_proportion <= 1
        ), "supervised_proportion must be in [0, 1]"

        self.dataset = dataset
        self.supervised_proportion = supervised_proportion
        self.labels_per_class = labels_per_class

        g = torch.Generator()
        g.manual_seed(seed)

        if labels_per_class is not None and labels_per_class > 0:
            print(
                f"Label per class has been set to {labels_per_class}, skipping supervised_proportion!"
            )

            # Get unique classes from the dataset
            # For HuggingFace datasets, access the underlying dataset
            if hasattr(self.dataset, "dataset"):
                dataset_obj = self.dataset.dataset
            else:
                dataset_obj = self.dataset

            classes = dataset_obj.unique(main_label_col)
            labels = torch.tensor(dataset_obj[main_label_col])
            supervised_split = torch.empty(
                (len(classes), labels_per_class), dtype=torch.long
            )

            for pt, c in enumerate(classes):
                idxs = torch.nonzero(labels == c, as_tuple=False).flatten()
                if len(idxs) < labels_per_class:
                    raise ValueError(
                        f"Not enough samples for class {c} got {len(idxs)} but need {labels_per_class}"
                    )

                # Randomly select labels_per_class samples
                selected = torch.randperm(len(idxs), generator=g)[:labels_per_class]
                supervised_split[pt] = idxs[selected]

            supervised_split = supervised_split.flatten()

        else:
            num_supervised_samples = int(len(self.dataset) * self.supervised_proportion)
            supervised_split = torch.randperm(len(self.dataset), generator=g)[
                :num_supervised_samples
            ]

        self.supervised_split = supervised_split.tolist()

        # Create binary column indicating whether the sample is supervised
        is_sup = torch.zeros(len(self.dataset), dtype=torch.bool)
        is_sup[supervised_split] = True

        # Add the is_supervised column to the underlying dataset
        if hasattr(self.dataset, "dataset"):
            # For HuggingFace datasets
            self.dataset.dataset = self.dataset.dataset.add_column(
                "is_supervised", is_sup.tolist()
            )
            self.is_supervised = is_sup.tolist() #TODO Debug this, not able to access is_supervised column in collate_fn

        else:
            # For other datasets, we'll store this information locally
            self.is_supervised = is_sup.tolist()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index] , self.is_supervised[index]


class SemiSupervisedSampler(torch.utils.data.DistributedSampler):
    """
    Sampler that ensures each batch contains a specific number of supervised samples.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        supervised_per_batch: int = 0,
        seed: int = 0,
    ):
        """
        Initialize the semi-supervised sampler.

        Args:
            dataset: The dataset (should be wrapped with SemiSupervisedWrapper)
            batch_size: Size of each batch
            supervised_per_batch: Number of supervised samples per batch
            seed: Random seed for reproducibility
        """
        self.dataset_len = len(dataset)

        indices = np.arange(len(dataset))
        self.num_batches = len(dataset) // batch_size
        self.sup_indices = np.array(dataset.supervised_split)
        self.unsup_indices = indices[~np.isin(indices, self.sup_indices)]
        self.batch_size = batch_size
        self.sup_per_batch = supervised_per_batch
        self.sup_ratio = dataset.supervised_proportion

        self.seed = seed
        self.epoch = 0

        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            if self.rank >= self.num_replicas or self.rank < 0:
                raise ValueError(
                    f"Invalid rank {self.rank}, rank should be in the interval [0, {self.num_replicas - 1}]"
                )
        else:
            self.num_replicas = 1
            self.rank = 0

        if self.dataset_len % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (self.dataset_len - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = self.dataset_len // self.num_replicas

    def __len__(self):
        return self.num_batches * self.batch_size

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        np.random.seed(self.seed + self.epoch)

        # Always sample with replacement if not enough supervised samples
        num_sup = len(self.sup_indices)
        num_unsup = len(self.unsup_indices)

        total_sup = self.num_batches * self.sup_per_batch
        total_unsup = self.num_batches * self.batch_size - total_sup

        if num_sup == 0 and self.sup_per_batch > 0:
            raise ValueError(
                "No supervised samples available, but supervised_per_batch is set to a non-zero value."
            )

        choose = np.random.choice

        # Select indices for supervised batch
        sup = choose(
            self.sup_indices,
            size=(self.num_batches, self.sup_per_batch),
            replace=num_sup < total_sup,
        )

        # Select indices for unsupervised batch
        unsup = choose(
            self.unsup_indices,
            size=(self.num_batches, self.batch_size - self.sup_per_batch),
            replace=num_unsup < total_unsup,
        )

        sup = torch.from_numpy(sup)
        unsup = torch.from_numpy(unsup)

        overall_slice = torch.cat([sup, unsup], 1).flatten()

        rank_slice = overall_slice[
            self.rank * self.num_samples : (self.rank + 1) * self.num_samples
        ]

        yield from rank_slice.tolist()

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.

        Args:
            epoch: The current epoch number
        """
        self.epoch = epoch

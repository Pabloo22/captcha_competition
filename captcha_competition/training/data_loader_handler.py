from typing import Optional

from torch.utils.data import DataLoader, IterableDataset, Dataset


class DataLoaderHandler:
    def __init__(
        self,
        dataset: Dataset | IterableDataset,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 4,
        steps_per_epoch: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.steps_per_epoch = steps_per_epoch
        self.loader = self._create_dataloader()

        self._iter_loader = None
        self._current_step = 0

    def _create_dataloader(self):
        # Determine if the dataset is an IterableDataset
        is_iterable = isinstance(self.dataset, IterableDataset)

        if is_iterable and self.steps_per_epoch is None:
            raise ValueError(
                "For IterableDataset, `steps_per_epoch` must be specified"
            )

        if not is_iterable and self.steps_per_epoch is not None:
            raise ValueError(
                "For non-IterableDataset, `steps_per_epoch` is not applicable"
            )

        # Create DataLoader with or without shuffling based on the dataset type
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            # Shuffle only for non-iterable datasets
            shuffle=not is_iterable and self.shuffle,
            num_workers=self.num_workers,
        )

    def __iter__(self):
        self._iter_loader = iter(self.loader)
        self._current_step = 0
        return self

    def __next__(self):
        if (
            self.steps_per_epoch is not None
            and self._current_step >= self.steps_per_epoch
        ):
            # If steps_per_epoch is defined and reached, stop iteration
            raise StopIteration
        self._current_step += 1
        return next(self._iter_loader)
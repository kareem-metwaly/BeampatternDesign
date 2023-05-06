"""The dataset class used to fetch files and read data + The dataset module
used for training/testing/validation splitting."""

import typing as t
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

# from scipy.io.matlab import savemat
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from utils.base_classes import TrainMode
from utils.config_classes import DatasetConfig, TrainConfig
from utils.item_classes import DatasetItem, DatasetItems
from utils.types import DirectoryPath

from datasets import DataLoaderRegistry, DatasetRegistry


class BaseDataset(Dataset):
    """Base Dataset that registers children to the registry."""

    configs: DatasetConfig

    def __init__(self, configs: DatasetConfig):
        """Specifies the configs, fetches the files from directory, and perform
        sanity check.

        :param configs: (DatasetConfig)
        """
        super().__init__()
        self.configs = configs

    def __init_subclass__(cls, **kwargs):
        """to store in the registry."""
        super().__init_subclass__(**kwargs)
        DatasetRegistry.register(cls)

    @abstractmethod
    def __getitem__(self, idx: int, mode: TrainMode = TrainMode.Unspecified) -> DatasetItem:
        """to retrieve a single sample from the dataset.

        :param idx: (int) the sample index to retrieve
        :param mode: (TrainMode) whether we are performing training, validation, or testing;
                                helpful if for example data augmentation would differ
        :return: (DatasetItem) One sample from the data
        """

    @abstractmethod
    def __len__(self) -> int:
        """returns the size of the dataset."""


class BaseDataLoader(LightningDataModule, ABC):
    """Base Dataset that registers children to the registry."""

    dataset_configs: DatasetConfig
    train_configs: TrainConfig
    _train_dataset: BaseDataset
    _val_dataset: BaseDataset
    _test_dataset: BaseDataset

    def __init__(self, dataset_configs: DatasetConfig, train_configs: TrainConfig, **kwargs):
        """storing configurations.

        :param dataset_configs: (DatasetConfig)
        :param train_configs: (TrainConfig)
        """
        super().__init__(**kwargs)
        self.dataset_configs = dataset_configs
        self.train_configs = train_configs
        self.__post_init__(**kwargs)

    def __init_subclass__(cls, **kwargs):
        """to store in the registry."""
        super().__init_subclass__(**kwargs)
        DataLoaderRegistry.register(cls)

    @abstractmethod
    def __post_init__(self, **kwargs):
        """for setting up splits of data, for example."""

    @abstractmethod
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """returns a pytorch dataloader for training."""

    @abstractmethod
    def val_dataloader(self) -> EVAL_DATALOADERS:
        """returns a pytorch dataloader for validation."""

    @abstractmethod
    def test_dataloader(self) -> EVAL_DATALOADERS:
        """returns a pytorch dataloader for testing."""

    @abstractmethod
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """returns a pytorch dataloader for prediction."""

    def dump_val_data(self, dir_path: DirectoryPath):
        assert hasattr(self, "_val_dataset") and self._val_dataset is not None
        dataset = DataLoader(self._val_dataset, collate_fn=DatasetItems.collate)
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        # for idx, sample in enumerate(tqdm(dataset, desc="dumping val data", position=-2, mininterval=100)):
        #     savemat(
        #         Path(path, str(idx) + ".mat"),
        #         {"desired_beampattern": sample.desired_beampattern.detach().cpu().numpy()},
        #     )
        with h5py.File(path.joinpath("data.h5"), "w") as file:
            for idx, sample in enumerate(tqdm(dataset, desc="dumping val data", position=-2, mininterval=100)):
                desired_beampattern = sample.desired_beampatterns.detach().squeeze().cpu().numpy()
                file.create_dataset(str(idx), desired_beampattern.shape, data=desired_beampattern)

        print("finished dumping data")


class BeamPatternDataModule(BaseDataLoader):
    """The data module that fetches elements for
    training/testing/validation."""

    train_val_test_splits: list[int]
    _train_dataloader: TRAIN_DATALOADERS
    _val_dataloader: EVAL_DATALOADERS
    _test_dataloader: EVAL_DATALOADERS
    _predict_dataloader: EVAL_DATALOADERS

    def __post_init__(self, **kwargs: t.Any) -> None:
        """setting up the splits for training/testing/validation."""
        super().__post_init__(**kwargs)
        self.train_val_test_splits = [0] * 3
        dataset = DatasetRegistry.get_class(self.dataset_configs.type)(configs=self.dataset_configs)
        self.train_val_test_splits[0] = round(len(dataset) * self.train_configs.train_val_test_split[0])
        self.train_val_test_splits[1] = round(len(dataset) * self.train_configs.train_val_test_split[1])
        self.train_val_test_splits[2] = len(dataset) - sum(self.train_val_test_splits[:-1])
        assert (
            self.train_val_test_splits[0] > 0
        ), f"We must have at least one sample to train with, given split is {self.train_val_test_splits}"
        if not self.train_val_test_splits[1]:
            warnings.warn("No validation is performed", category=UserWarning)
            if self.train_configs.train_val_test_split[1]:
                warnings.warn(
                    "The provided value is small enough to not be able to take at least one sample from the dataset",
                    category=UserWarning,
                )
        if not self.train_val_test_splits[2]:
            warnings.warn("No testing is performed", category=UserWarning)
            if self.train_configs.train_val_test_split[2]:
                warnings.warn(
                    "The provided value is small enough to not be able to take at least one sample from the dataset",
                    category=UserWarning,
                )

        generator = torch.Generator()
        if self.train_configs.random_seed:
            generator.manual_seed(self.train_configs.random_seed)

        self._train_dataset, self._val_dataset, self._test_dataset = random_split(
            dataset, self.train_val_test_splits, generator
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Creates the training dataloader if not already defined."""
        if not hasattr(self, "_train_dataloader"):
            self._train_dataloader = DataLoader(
                self._train_dataset,
                batch_size=self.train_configs.batch_size,
                shuffle=True,
                num_workers=self.train_configs.num_workers,
                collate_fn=DatasetItems.collate,
            )
        return self._train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Creates the validation dataloader if not already defined."""
        if not hasattr(self, "_val_dataloader"):
            self._val_dataloader = DataLoader(
                self._val_dataset,
                batch_size=self.train_configs.batch_size,
                shuffle=False,
                num_workers=self.train_configs.num_workers,
                collate_fn=DatasetItems.collate,
            )
        return self._val_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Creates the testing dataloader if not already defined."""
        if not hasattr(self, "_test_dataloader"):
            self._test_dataloader = (
                DataLoader(
                    self._test_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=self.train_configs.num_workers,
                    collate_fn=DatasetItems.collate,
                )
                if len(self._test_dataset)
                else self.val_dataloader()
            )
        return self._test_dataloader

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Creates the prediction dataloader if not already defined."""
        if not hasattr(self, "_predict_dataloader"):
            self._predict_dataloader = DataLoader(
                self._test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                collate_fn=DatasetItems.collate,
            )
        return self._predict_dataloader

    def teardown(self, stage: t.Optional[str] = None) -> None:
        """used for destroying any object after finishing usage of these
        dataloader."""
        pass

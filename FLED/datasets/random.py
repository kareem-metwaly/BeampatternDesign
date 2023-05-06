"""Randomly generate beampatterns without a corresponding groundtruth."""
import math
import typing as t

import torch
from datasets.base_dataset import BaseDataset
from utils.base_classes import TrainMode
from utils.config_classes import DatasetConfig
from utils.item_classes import DatasetItem
from utils.types import FilePath


class RandomBeamPattern(BaseDataset):
    """This is a PyTorch implementation to work in training to retrieve data.

    The Beampattern is generated randomly according to configs.
     - `OptimizedWaveform` contains the result of the PDR algorithm or an empty tensor
     - `DesiredBeampattern` contains the input desired beampattern
    """

    files: list[FilePath]

    def __init__(self, configs: DatasetConfig):
        """Specifies the configs, fetches the files from directory, and perform
        sanity check.

        :param configs: (DatasetConfig)
        """
        super().__init__(configs=configs)
        self.resolution_K = math.floor(configs.params.K / 4)
        self.resolution_N = math.floor(configs.params.N / 4)
        self.stride_K = math.floor(self.resolution_K)
        self.stride_N = math.floor(self.resolution_N)
        self.size_K = math.ceil((configs.params.K - self.resolution_K) / self.stride_K) + 1
        self.size_N = math.ceil((configs.params.N - self.resolution_N) / self.stride_N) + 1
        self._len = 2 ** (self.size_K * self.size_N)
        if configs.max_length and self._len > configs.max_length:
            self._range = torch.randperm(self._len)[: configs.max_length]
            self._len = configs.max_length

    def __getitem__(self, index: int, mode: TrainMode = TrainMode.Unspecified) -> t.Optional[DatasetItem]:
        """Gets one sample from the dataset without performing sanity check."""
        desired_beampattern = torch.zeros((self.configs.params.K, self.configs.params.N))
        if hasattr(self, "_range"):
            index = self._range[index]
        active_bits = reversed(bin(index))  # LSB first
        for idx, bit in enumerate(active_bits):
            if bit == "1":
                id_N = idx % self.size_N
                id_K = math.floor(idx / self.size_N)
                K_start = id_K * self.stride_K
                N_start = id_N * self.stride_N
                desired_beampattern[
                    K_start : K_start + self.resolution_K, N_start : N_start + self.resolution_N
                ] = self.configs.params.N
            elif bit == "b":
                break
        sample = self.preprocess_sample(
            DatasetItem(
                optimum_waveform=torch.tensor([[]]), desired_beampattern=desired_beampattern.to(torch.complex64)
            )
        )
        return sample

    def __len__(self) -> int:
        """the size of the dataset."""
        return self._len

    def preprocess_sample(self, dataset_item: DatasetItem) -> DatasetItem:
        """we may need to perform some preprocessing such as normalization of
        the range of values."""
        max_val = self.configs.params.N  # desired beampattern has min value of 0 and max of N
        dataset_item.desired_beampattern /= max_val  # now it is in the range of 0, 1
        return dataset_item

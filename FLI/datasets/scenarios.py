"""Specific scenarios for validation to generate beampatterns without a
corresponding groundtruth."""
import typing as t

import torch
from datasets.base_dataset import BaseDataset
from utils.base_classes import TrainMode
from utils.config_classes import DatasetConfig
from utils.item_classes import DatasetItem
from utils.types import FilePath


class ScenariosBeamPattern(BaseDataset):
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
        self.params = configs.params
        self.n_scenarios = 10
        self.fc = configs.params.fc
        self.B = configs.params.B
        self.K = torch.tensor(configs.params.K)
        self.N = torch.tensor(configs.params.N)
        self.start_freq = self.fc - self.B / 2.0
        self.end_freq = self.fc + self.B / 2.0
        self.start_angle = 0
        self.end_angle = 180
        self.scenarios = [
            [
                [self.start_freq, self.end_freq, 88, 132, self.N],
            ],
            [
                [self.start_freq, self.fc, self.start_angle, 44, self.N],
                [self.start_freq, self.fc, 88, self.end_angle, self.N],
                [self.fc, self.end_freq, self.start_angle, 132, self.N],
                [self.fc, self.end_freq, 176, self.end_angle, self.N],
            ],
            [
                [self.start_freq, 943.75e6, self.start_angle, self.end_angle, self.N],
                [1e9, self.end_freq, self.start_angle, self.end_angle, self.N],
                [943.75e6, 1e9, self.start_angle, 44, self.N],
                [943.75e6, 1e9, 88, 132, self.N],
                [943.75e6, 1e9, 176, self.end_angle, self.N],
                [981.25e6, 1e9, 44, 88, self.N],
                [943.75e6, 942.5e6, 132, 176, self.N],
            ],
        ]
        # list of scenarios param, each scenario consists of a list of tuples (length 5).
        # Each tuple has (start_f, end_f, start_angle, end_angle, value)

    def __getitem__(self, index: int, mode: TrainMode = TrainMode.Unspecified) -> t.Optional[DatasetItem]:
        """Gets one sample from the dataset without performing sanity check."""
        desired_beampattern = torch.zeros((self.K, self.N))
        freq_range = self.end_freq - self.start_freq
        angle_range = self.end_angle - self.start_angle
        for f_start, f_end, ang_start, ang_end, value in self.scenarios[index]:
            K_start = torch.clamp(
                torch.round((ang_start - self.start_angle) / angle_range * self.K), 0, self.K - 1
            ).int()
            K_end = (
                torch.clamp(torch.round((ang_end - self.start_angle) / angle_range * self.K), 0, self.K - 1).int() + 1
            )
            N_start = torch.clamp(torch.round((f_start - self.start_freq) / freq_range * self.N), 0, self.N - 1).int()
            N_end = torch.clamp(torch.round((f_end - self.start_freq) / freq_range * self.N), 0, self.N - 1).int() + 1
            desired_beampattern[K_start:K_end, N_start:N_end] = value

        sample = self.preprocess_sample(
            DatasetItem(
                optimum_waveform=torch.tensor([[]]), desired_beampattern=desired_beampattern.to(torch.complex64)
            )
        )
        return sample

    def __len__(self) -> int:
        """the size of the dataset."""
        return len(self.scenarios)

    def preprocess_sample(self, dataset_item: DatasetItem) -> DatasetItem:
        """we may need to perform some preprocessing such as normalization of
        the range of values."""
        max_val = self.configs.params.N  # desired beampattern has min value of 0 and max of N
        dataset_item.desired_beampattern /= max_val  # now it is in the range of 0, 1
        return dataset_item

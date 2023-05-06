"""contains the implementation of the initial module that takes the desired
beampattern and estimates good initial guesses for the waveforms."""
import typing as t

from torch import nn
from utils.build_modules import build_fc_network
from utils.config_classes import FullyConnectedConfig
from utils.types import Tensor1D, Tensor2D


class Initial(nn.Module):
    """implementation of the initial submodule of the network."""

    model: nn.Module
    n_branches: int

    def __init__(self, configs: t.Sequence[FullyConnectedConfig], n_branches: int):
        """stores the configurations and builds the submodule which is a list
        of linear networks.

        :param configs: (Sequence[FullyConnectedConfig)
        :param n_branches: (int)
        """
        super().__init__()
        self.n_branches = n_branches
        configs[-1].out_channels *= self.n_branches
        self.model = build_fc_network(configs=configs).Model

    def forward(self, x: Tensor2D) -> Tensor1D:
        """loop through different modules in the list and stack the results,

        :param x: the desired beam pattern
        :return: n different generated initial waveforms
        """
        x = x.flatten(1)  # keeps the batch dim unchanged and flatten the 2d to 1d
        B, _ = x.size()
        return self.model(x).view(B, self.n_branches, -1)

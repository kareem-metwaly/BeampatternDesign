"""contains implementation of a single unrolled step."""

import typing as t

import torch
from torch import nn
from utils.activations import Magnitude
from utils.base_classes import Criterion
from utils.build_modules import (
    BuildOutput,
    FixedPrune,
    Project,
    Retract,
    TrainableExpand,
    TrainablePrune,
    build_fc_network,
)
from utils.config_classes import ModelConfig
from utils.item_classes import StepOutput
from utils.types import Tensor1D, Tensor2D


class UnrolledStep(nn.Module):
    """implementations of a single step [Direction Evaluation, Project,
    Retract, Prune, Expand]"""

    trainable_prune: bool  # whether the prune network is trainable or not
    MN: int  # Product of M*N the input size (the waveform size)
    n_expand: int  # how many remaining after expansion (should match the second dimension of the input)

    def __init__(self, configs: ModelConfig, min_criterion: t.Optional[Criterion] = None):
        """Creates a new step of the unrolled algorithm.

        :param configs: (ModelConfig) configuration of the model
        :param min_criterion: (Optional[Criterion], default: None)
                                    the criterion used to minimize; used by the prune module (if fixed)
        """
        super().__init__()
        assert bool(configs.trainable_prune) ^ bool(min_criterion), (
            f"we should either define min_criterion or use a trainable prune network, "
            f"given: {min_criterion} and {configs.trainable_prune}"
        )
        self.direction_evaluation = build_fc_network(configs.direction_evaluation).Model
        self.step_evaluation = nn.Sequential(
            nn.Linear(
                in_features=configs.direction_evaluation[-1].out_channels + configs.direction_evaluation[0].in_channels,
                out_features=50,
                bias=False,
                dtype=torch.complex64,
            ),
            Magnitude(),
            nn.BatchNorm1d(num_features=50, affine=True),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=50,
                out_features=1,
                bias=True,
            ),
            nn.LeakyReLU(),
        )
        self.project = Project()
        self.retract = Retract(epsilon=configs.epsilon)

        self.prune = (
            TrainablePrune(configs=configs.prune_params, output_size=configs.n_prune)
            if configs.trainable_prune
            else FixedPrune(output_size=configs.n_prune, min_criterion=min_criterion)
        )
        self.expand = TrainableExpand(configs.expand, output_size=configs.n_expand)
        self.trainable_prune = configs.trainable_prune
        # some values that are needed in the forwarded propagation
        self.n_expand = configs.n_expand
        self.MN = configs.direction_evaluation[0].in_channels

    def forward(self, x: Tensor1D, desired: t.Optional[Tensor2D] = None) -> StepOutput:
        """forward propagation of a single step."""
        assert desired is not None or self.trainable_prune, (
            f"We should either have the desired beampattern or use a trainable prune network, "
            f"given: {desired} and {self.trainable_prune}"
        )

        # combining the batch dimension and the number of trials;
        # as we treat each one independently in the following stages
        x = x.view(-1, self.MN)
        eta = self.direction_evaluation(x)
        beta = self.step_evaluation(torch.cat((x, eta), dim=-1))
        x = self.project(x, eta=eta, beta=beta)
        x = self.retract(x)

        # going back to the same size B x Ne x M*N;
        # as the following layers depends on the number of trials to prune/expand
        x = x.view(-1, self.n_expand, self.MN)
        # x = self.prune(x, desired=desired)
        # x = self.expand(x)
        return StepOutput(
            estimated_gradients=eta.view(-1, self.n_expand, self.MN),
            estimated_waveforms=x,
            estimated_steps=beta.view(-1, self.n_expand),
        )


def build_unrolled_step(
    configs: ModelConfig, min_criterion: t.Optional[Criterion] = None, test: bool = True
) -> BuildOutput:
    """Creates a new step of the unrolled algorithm.

    :param configs: (ModelConfig) configuration of the model
    :param min_criterion: (Optional[Criterion], default: None) the criterion used to minimize; used by the prune module
                                                                (if fixed)
    :param test: (bool) whether to validate the dimensions of the sequence is matching or not
    :return: (BuildOutput) a module output that represent the whole step
    """

    model = UnrolledStep(configs=configs, min_criterion=min_criterion)

    if test:
        inp = torch.rand([2, configs.n_expand, configs.direction_evaluation[0].in_channels], dtype=torch.complex64)
        desired = torch.rand([2, configs.initial[0].in_channels])
        out = model(inp, desired=desired)
        assert out.shape == inp.shape, (
            f"Unrolled step shouldn't change the dims of the input, it only alters vals, "
            f"given: {inp.shape}, and output: {out.shape}"
        )
        return BuildOutput(model, out.shape)
    return BuildOutput(model, None)

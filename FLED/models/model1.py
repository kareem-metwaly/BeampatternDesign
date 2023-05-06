"""First model to be trained and tested."""
import typing as t

from losses.objective import Objective
from models.base_model import Model
from models.initial import Initial
from models.unrolled_step import UnrolledStep
from torch import nn
from utils.base_classes import TrainMode
from utils.build_modules import FixedPrune
from utils.item_classes import DatasetItems, FinalModelOutput
from utils.types import Tensor2D


class Model1(Model):
    """First model to train, contains initial, steps and final arg_min."""

    def _define_modules(self, **kwargs) -> None:
        """builds initial block and several unrolled steps. At the end, argmin
        module based on pruning architecture.

        :param kwargs:
        :return: None
        """
        Ne = self.model_configs.n_expand
        criterion = (
            None
            if self.model_configs.trainable_prune
            else Objective(is_criterion=True, loss_configs=None, dataset_configs=self.dataset_configs)
        )
        self.initial_module = Initial(configs=self.model_configs.initial, n_branches=Ne)
        if self.model_configs.shared_weights:
            step = UnrolledStep(configs=self.model_configs, min_criterion=criterion)
            self.steps_modules = nn.ModuleList([step] * self.model_configs.n_steps)
        else:
            self.steps_modules = nn.ModuleList(
                [
                    UnrolledStep(configs=self.model_configs, min_criterion=criterion)
                    for _ in range(self.model_configs.n_steps)
                ]
            )

        # the last prune: removes all and keeps the best one
        self.arg_min_module = FixedPrune(
            output_size=1,
            min_criterion=Objective(is_criterion=True, loss_configs=None, dataset_configs=self.dataset_configs),
        )
        self.avg_min_module = lambda x: x.mean(dim=1)

    def forward(self, inp: t.Union[Tensor2D, DatasetItems], **kwargs) -> FinalModelOutput:
        """performs forward propagation for training.

        :param inp:
        :param kwargs:sum
        :return:
        """
        compute_loss = kwargs.get("compute_loss", True)
        if isinstance(inp, DatasetItems):
            inp_desired = inp.desired_beampatterns
            mode = TrainMode.Train if self.training else TrainMode.Validate
        else:
            inp_desired = inp
            mode = TrainMode.Test

        initial_output = steps_output = self.initial_module(inp_desired)  # x0 of size B x Ne x MN

        for i, step in enumerate(self.steps_modules):
            steps_output = step(steps_output, desired=inp_desired)  # x of size B x Ne x MN
            if compute_loss and mode in [TrainMode.Train, TrainMode.Validate]:
                if self.grad_loss is not None:
                    self.add_loss_hook(self.grad_loss(inp, steps_output, mode, step_id=i))
                if self.disparity_loss is not None:
                    epoch = kwargs["epoch"] if "epoch" in kwargs else None
                    self.add_loss_hook(self.disparity_loss(inp, steps_output, mode, epoch=epoch, step_id=i))

            steps_output = steps_output.estimated_waveforms

        if mode is TrainMode.Train:
            final_x = self.avg_min_module(steps_output)
        else:
            final_x = self.arg_min_module(x=steps_output, desired=inp_desired).squeeze(dim=1)

        if self._running_torchscript:
            # if we are logging the graph only
            return final_x

        return FinalModelOutput(estimated_waveforms=final_x, initial=initial_output, steps=steps_output)

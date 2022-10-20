"""Contains the module for calculating the objective function of the
optimization problem."""
import typing as t

import torch
from losses.base_loss import LossModule
from utils.base_classes import TrainMode
from utils.build_modules import Scale
from utils.config_classes import LossConfig, ScaleConfig
from utils.item_classes import DatasetItems, LossItem, ModelOutput, TotalLoss
from utils.physics import calculate_diff_dB, find_matrix_H_using_params
from utils.types import Tensor1D, Tensor2D, TensorScaler


class Objective(LossModule):
    """Given an input Tensor representing Beampattern and a desired Tensor
    representing Waveform, calculated the error takes H (the product of A and F
    matrices) in the initialization when invoked (takes DatasetItems and
    ModelOutput) and returns (TotalLoss)"""

    H: torch.Tensor
    is_criterion: bool

    def __init__(self, loss_configs: t.Optional[LossConfig] = None, **kwargs):
        """sets the value of H that will be used later in the loss
        calculations.

        :param kwargs: arbitrary
        """
        super().__init__(loss_configs=loss_configs, **kwargs)
        if "dataset_configs" in kwargs:
            self.dataset_configs = kwargs["dataset_configs"]
            self.H = find_matrix_H_using_params(self.dataset_configs.params)
        elif "H" in kwargs:
            self.H = kwargs["H"]
        else:
            raise ValueError("Either dataset_configs or H must be given")
        # self.H = torch.tensor(self.H, requires_grad=False)
        self.H = torch.nn.Parameter(self.H, requires_grad=False)
        self.scale = Scale(ScaleConfig(inp_min=0, inp_max=self.dataset_configs.params.N, out_min=0, out_max=1))

    def complete_forward(
        self, input_batch: DatasetItems, model_output: ModelOutput, mode: TrainMode = TrainMode.Unspecified, **kwargs
    ) -> TotalLoss:
        """calls loss function, but first prepares the data.

        :param input_batch:
        :param model_output:
        :param mode:
        :param kwargs:
        :return:
        """
        desired = input_batch.desired_beampatterns
        x = model_output.estimated_waveforms.unsqueeze(dim=1)
        return TotalLoss(
            LossItem(
                name=mode.value + "/OptimizationObjective",
                value=self.core_forward(x=x, desired=desired),
                isLogged=True,
            )
        )

    def core_forward(
        self, x: Tensor1D, desired: Tensor2D, reduce: t.Optional[bool] = True
    ) -> t.Union[Tensor1D, TensorScaler]:
        """calculates the objective function of the optimization problem.

        :param x: Tensor1D of size B x Ne x MN
        :param desired: Tensor2D of size B x K x N, assumed to be normalized from 0 to 1
        :param reduce: (bool, default: True) whether to sum the values to a single scaler or not
        :return: a scaler tensor representing f(x)
        """
        # we have to unsqueeze desired as x has an extra dim Ne
        loss = (desired.unsqueeze(dim=1) - self.estimated_beampattern(x, normalize=True)).abs().pow(2)
        if reduce:
            return loss.mean()
        return loss

    def estimated_beampattern(self, x: Tensor1D, normalize: bool = True) -> Tensor2D:
        """estimates the generated beampattern based on the input (which could
        be a batch of input)

        :param x: Tensor1D of size B x Ne x MN
        :param normalize: bool; default=True, set the range of the output to be between [0, 1]
        :return: a Tensor2D representing the beampattern of size B x K x N
        """
        if x.dtype != self.H.dtype:  # provided x may not be complex be of the same type as H
            x = x.to(self.H.dtype)
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1)  # as it must be of size B x Ne x MN (sometimes we may input B x MN incorrectly)
        out = torch.tensordot(x, self.H, dims=([2], [2])).abs()
        if normalize:
            out = self.scale(out)
        return out

    def estimate_db_diff(self, x: Tensor1D, desired: Tensor2D) -> Tensor1D:
        """calculates the objective function of the optimization problem.

        :param Tensor1D x: of size B x Ne x MN or B x MN
        :param Tensor2D desired: of size B x K x N, assumed to be normalized from 0 to 1
        :return: Tensor1D a scaler tensor representing error in dB
        """
        if len(x.size()) == 2:
            x = x.unsqueeze(dim=1)
        predictions = self.estimated_beampattern(x, normalize=True)
        return calculate_diff_dB(predictions=predictions.mean(dim=1), groundtruths=desired)

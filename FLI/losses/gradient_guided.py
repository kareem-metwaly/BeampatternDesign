"""Contains the module for calculating the gradient guided loss function."""
import typing as t

import torch
from losses.base_loss import LossModule
from torch.nn import Parameter
from utils.base_classes import TrainMode
from utils.config_classes import LossConfig
from utils.item_classes import DatasetItems, LossItem, StepOutput, TotalLoss
from utils.physics import find_matrix_H_using_params
from utils.types import Tensor1D, Tensor2D, TensorScaler


class GradientGuidedLoss(LossModule):
    """Given an input Tensor representing the estimated gradient direction and
    a desired Tensor representing Waveform, calculated the error takes H (the
    product of A and F matrices) in the initialization when invoked (takes
    DatasetItems and ModelOutput) and returns (TotalLoss)"""

    H: Parameter
    is_criterion: bool

    def __init__(self, loss_configs: t.Optional[LossConfig] = None, **kwargs):
        """sets the value of P and q that will be used later in the loss
        calculations.

        :param kwargs: arbitrary; may have P and q values, or else it will be calculated from the parameters
        """
        assert "dataset_configs" in kwargs or "H" in kwargs, "Either dataset_configs or H must be given"
        super().__init__(loss_configs=loss_configs, **kwargs)
        if "dataset_configs" in kwargs:
            self.dataset_configs = kwargs["dataset_configs"]
            self.H = find_matrix_H_using_params(self.dataset_configs.params)
        else:
            self.H = kwargs["H"]
        self.HH = torch.tensordot(self.H, self.H.conj(), dims=((0, 1), (0, 1)))
        self.H = Parameter(self.H, requires_grad=False)  # size K x N x MN
        self.HH = Parameter(self.HH, requires_grad=False)  # size MN x MN

    def complete_forward(
        self,
        input_batch: DatasetItems,
        model_output: StepOutput,
        mode: TrainMode = TrainMode.Unspecified,
        **kwargs,
    ) -> TotalLoss:
        """calls loss function, but first prepares the data.

        :param input_batch:
        :param model_output:
        :param mode:
        :param kwargs:
        :return:
        """
        eta = model_output.estimated_gradients
        x = model_output.estimated_waveforms
        desired = input_batch.desired_beampatterns
        step_id = kwargs["step_id"] if "step_id" in kwargs else ""
        return TotalLoss(
            LossItem(
                name=mode.value + f"/GradientGuided({step_id})",
                value=self.core_forward(x=x, eta=eta, desired=desired),
                weight=self.weight,
                isLogged=True,
            )
        )

    def core_forward(
        self, x: Tensor1D, eta: Tensor1D, desired: Tensor2D, reduce: t.Optional[bool] = True
    ) -> t.Union[Tensor1D, TensorScaler]:
        """calculates the gradient deviation loss.

        :param x: Tensor1D of size B x Ne x MN representing the estimated waveforms
        :param eta: Tensor1D of size B x Ne x MN representing the estimated gradients
        :param desired: Tensor2D of size B x K x N, assumed to be normalized from 0 to 1
        :param reduce: (bool, default: True) whether to sum the values to a single scaler or not
        :return: a scaler tensor representing (Mean)-Square-Error between the expected gradient and the estimated one
        """
        gt_eta = self.compute_gradient_descent(x, desired)
        loss = (gt_eta - eta).abs().pow(2)
        if reduce:
            return loss.mean()
        return loss

    def compute_gradient_descent(self, x: Tensor1D, desired: Tensor2D):
        """computes the gradient using the lemma in the paper.

        :param x: Tensor1D of size B x Ne x MN representing the estimated waveforms
        :param desired: Tensor2D of size B x K x N, assumed to be normalized from 0 to 1
        :return: a vector tensor representing the gradient descent direction
        """
        Hx = torch.tensordot(x, self.H, dims=((2,), (2,)))
        return torch.tensordot(
            desired.unsqueeze(1) * Hx.conj() / Hx.abs(), self.H, dims=((2, 3), (0, 1))
        ) - torch.tensordot(x.conj(), self.HH, dims=((2,), (1,)))

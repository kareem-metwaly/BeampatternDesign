"""Contains the module for calculating the MSELoss between estimated waveform
and the GT."""
import typing as t

from losses.base_loss import LossModule
from utils.base_classes import TrainMode
from utils.item_classes import DatasetItems, LossItem, ModelOutput, TotalLoss
from utils.types import Tensor1D, TensorScaler


class MSELoss(LossModule):
    """takes an input Tensor representing the estimated waveform and a desired
    Tensor representing Waveform, when invoked (takes DatasetItems and
    ModelOutput) and returns (TotalLoss)"""

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
        desired = input_batch.optimum_waveforms
        x = model_output.estimated_waveforms
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1)
        return TotalLoss(
            LossItem(
                name=mode.value + "/OptimizationObjective",
                value=self.core_forward(x=x, desired=desired),
                isLogged=True,
            )
        )

    def core_forward(
        self, x: Tensor1D, desired: Tensor1D, reduce: t.Optional[bool] = True
    ) -> t.Union[Tensor1D, TensorScaler]:
        """calculates the loss.

        :param x: Tensor1D of size B x Ne x MN
        :param desired: Tensor1D of size B x MN
        :param reduce: (bool, default: True) whether to sum the values to a single scaler or not
        :return: a tensor representing f(x)
        """
        loss = (x - desired.unsqueeze(dim=1)).abs().pow(2)
        if reduce:
            loss = loss.mean()
        return loss

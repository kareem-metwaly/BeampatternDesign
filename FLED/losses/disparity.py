"""Contains the module for calculating the disparity loss term between
different generated waveforms."""
import typing as t

from losses.base_loss import LossModule
from torch.nn import Parameter
from utils.base_classes import TrainMode
from utils.complex_functions import complex_l2_norm
from utils.item_classes import DatasetItems, LossItem, ModelOutput, TotalLoss
from utils.types import Tensor1D, TensorScaler


class DisparityLoss(LossModule):
    """takes an input Tensor representing a collection of estimated waveform
    and computes a disparity loss based on the absolute value of the inner
    product (identical to absolute value of Cosine Similarity if inputs have
    unit complex lengths)"""

    def complete_forward(
        self, input_batch: DatasetItems, model_output: ModelOutput, mode: TrainMode = TrainMode.Unspecified, **kwargs
    ) -> TotalLoss:
        """calls loss function, but first prepares the data.

        :param DatasetItems input_batch:
        :param ModelOutput model_output:
        :param TrainMode mode:
        :param kwargs:
        :return: TotalLoss
        """
        if isinstance(self.weight, Parameter):
            weight = kwargs["weight"]
        else:
            assert "epoch" in kwargs
            weight = self.weight(kwargs["epoch"])

        x = model_output.estimated_waveforms
        assert len(x.shape) == 3, "the estimated waveforms should have"
        step_id = kwargs["step_id"] if "step_id" in kwargs else ""
        return TotalLoss(
            LossItem(
                name=mode.value + f"/Disparity{step_id}",
                value=self.core_forward(x=x, desired=None),
                isLogged=True,
                weight=weight,
            )
        )

    def core_forward(
        self, x: Tensor1D, desired: t.Literal[None] = None, reduce: t.Optional[bool] = True
    ) -> t.Union[Tensor1D, TensorScaler]:
        """calculates the disparity loss for the input x.

        :param x: Tensor1D of size B x Ne x MN
        :param desired: should be None (kept for compatibility)
        :param reduce: (bool, default: True) whether to sum the values to a single scaler or not
        :return: a tensor representing ~ CosSim(x)
        """
        desired = x.mean(dim=1, keepdim=True)  # of size B x 1 x MN
        desired = desired.div(complex_l2_norm(desired, keepdim=True).clamp(min=1e-5))
        loss = x.conj().mul(desired).real.sum(dim=[1, 2]).abs()
        if reduce:
            loss = loss.mean()
        return loss

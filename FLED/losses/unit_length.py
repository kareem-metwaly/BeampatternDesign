"""Contains the module for calculating the unit length regularization loss term
for the output of each step."""
import typing as t

from losses.base_loss import LossModule
from utils.base_classes import TrainMode
from utils.complex_functions import complex_cosine_similarity
from utils.item_classes import DatasetItems, LossItem, ModelOutput, TotalLoss
from utils.types import Tensor1D, TensorScaler


class UnitLengthLoss(LossModule):
    """takes an input Tensor representing a collection of estimated waveform
    and computes the distance between the waveform and its unit length loss
    based on the absolute value of the inner product (identical to absolute
    value of Cosine Similarity if inputs have unit complex lengths)"""

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
        x = model_output.estimated_waveforms
        assert len(x.shape) == 3, "the estimated waveforms should have"
        step_id = kwargs["step_id"] if "step_id" in kwargs else ""
        return TotalLoss(
            LossItem(
                name=mode.value + f"/Disparity{step_id}",
                value=self.core_forward(x=x, desired=None),
                isLogged=True,
                weight=self.weight,
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
        # B, Ne, MN = x.size()
        # desired = desired.repeat(1, Ne, 1).view(-1, MN)
        # x = x.view(-1, MN)
        cosine = complex_cosine_similarity(x, desired, dims=([2], [2]))
        loss = cosine.real.clamp(min=0) + cosine.imag.abs()
        if reduce:
            loss = loss.mean()
        return loss

"""implements a loss function that builds multiple loss terms bases on
configs."""

import typing as t

from losses import LossRegistry
from losses.base_loss import LossModule
from torch import Tensor
from torch.nn import ModuleDict, Parameter
from utils.base_classes import TrainMode
from utils.config_classes import LossConfig, LossTermTrueType
from utils.item_classes import DatasetItems, FinalModelOutput, LossItem, TotalLoss
from utils.types import Tensor1D, Tensor2D, TensorScaler


class BuildLosses(LossModule):
    """based on the configs it builds multiple loss terms."""

    def __init__(self, loss_configs: t.Optional[LossConfig] = None, **kwargs):
        """sets the value of H that will be used later in the loss
        calculations.

        :param kwargs: arbitrary
        """
        super().__init__(loss_configs=loss_configs, **kwargs)
        self.weights = {}
        self.gt = {}
        self.loss_modules = ModuleDict()
        self.names = {}
        for loss_item in self.loss_configs.outputs:
            key = loss_item.name + "/" + loss_item.type
            self.loss_modules.update({key: LossRegistry.get_class(loss_item.type)(loss_configs, **kwargs)})
            self.weights[key] = Parameter(Tensor([loss_item.weight]), requires_grad=False)
            self.gt[key] = loss_item.gt
            self.names[key] = loss_item.name

    def complete_forward(
        self,
        input_batch: DatasetItems,
        model_output: FinalModelOutput,
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
        desired_beampattern = input_batch.desired_beampatterns
        desired_waveform = input_batch.optimum_waveforms
        loss_items = []
        for key in self.loss_modules.keys():
            loss_items.append(
                LossItem(
                    name=key,
                    value=self.loss_modules[key].core_forward(
                        x=model_output.__getattribute__(self.names[key]),
                        desired=desired_waveform if self.gt[key] is LossTermTrueType.Waveform else desired_beampattern,
                    ),
                    isLogged=True,
                    weight=self.weights[key],
                )
            )
        return TotalLoss(loss_items)

    def core_forward(
        self, x: Tensor1D, desired: Tensor2D, reduce: t.Optional[bool] = True
    ) -> t.Union[Tensor1D, TensorScaler]:
        """does nothing as we use the core_forward of submodules."""

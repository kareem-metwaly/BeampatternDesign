"""Contains Base Class for Loss Modules."""

import typing as t
from abc import ABC, abstractmethod

from losses import LossRegistry
from torch import Tensor
from torch.nn import Module, Parameter
from utils.base_classes import TrainMode
from utils.config_classes import LossConfig, LossWeightConfig
from utils.item_classes import DatasetItems, ModelOutput, TotalLoss
from utils.weight_functions import WeightRegistry


class LossModule(Module, ABC):
    """Base Class for Loss Modules, any loss module must inherit from it, and
    it must define forward and loss methods.

    :param loss_configs: (LossConfig) configuration of the loss function
    :param is_criterion: (bool) whether to use loss or forward method in calling the module
    """

    loss_configs: t.Optional[LossConfig]
    weight: t.Optional[t.Union[Parameter, t.Callable]]

    def __init__(self, loss_configs: t.Optional[LossConfig], is_criterion: bool = False, **kwargs):
        """sets is_criterion to either call complete_forward (taking data
        structures) or core_forward (taking actual operating elements
        explicitly)."""
        super().__init__()
        self.loss_configs = loss_configs
        if is_criterion:
            self.forward = self.core_forward
        else:
            self.forward = self.complete_forward
        self.weight = None
        if "weight" in kwargs:
            self.weight = kwargs["weight"]
            if isinstance(self.weight, float):
                self.weight = Parameter(Tensor([self.weight]), requires_grad=False)
            elif isinstance(self.weight, LossWeightConfig):
                self.weight = WeightRegistry.get_method(self.weight)
            else:
                raise NotImplementedError(f"unsupported weight type, given: {LossWeightConfig}")

    def __init_subclass__(cls, **kwargs):
        """Invoked to register all loss modules in the LossRegistry."""
        super().__init_subclass__(**kwargs)
        LossRegistry.register(cls)

    @abstractmethod
    def complete_forward(
        self, input_batch: DatasetItems, model_output: ModelOutput, mode: TrainMode = TrainMode.Unspecified, **kwargs
    ) -> TotalLoss:
        """Implements the forward propagation through training, validation or
        test."""
        ...

    @abstractmethod
    def core_forward(self, *args, **kwargs) -> t.Any:
        """calculates the loss depending on its actual required arguments,
        forward method will probably call this method after preparing the data
        for its consumption and then prepares a TotalLoss output."""

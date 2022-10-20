"""defines the basic model that every model must inherit from Mainly, any model
must define two methods forward and _define_modules."""
import typing as t
import warnings
from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
from losses.base_loss import LossModule, LossRegistry
from losses.objective import Objective
from models import ModelRegistry
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.base_classes import TrainMode
from utils.config_classes import (
    DatasetConfig,
    LossConfig,
    LRSchedulerType,
    ModelConfig,
    OptimizerType,
    TrainConfig,
    TrainPhaseOption,
)
from utils.etc import freeze_module, get_cpu_mem, get_num_of_tensors, unfreeze_module
from utils.item_classes import DatasetItems, ModelOutput, TotalLoss


class Model(pl.LightningModule, ABC):
    """The basic model structure."""

    model_configs: ModelConfig
    train_configs: TrainConfig
    dataset_configs: DatasetConfig
    loss_configs: LossConfig
    loss_module: LossModule
    current_train_phase: int

    # used to add some loss terms internally inside models (for regularization for example)
    _registered_loss_hooks: t.Optional[TotalLoss] = None

    def __init__(
        self,
        model_configs: ModelConfig,
        train_configs: TrainConfig,
        loss_configs: LossConfig,
        dataset_configs: DatasetConfig,
        **kwargs,
    ):
        """stores the configs and calls the _define_modules method It
        constructs the loss module as well.

        :param model_configs:
        :param train_configs:
        :param loss_configs:
        :param dataset_configs:
        :param kwargs:
        """
        super().__init__()
        self.model_configs = model_configs
        self.train_configs = train_configs
        self.loss_configs = loss_configs
        self.dataset_configs = dataset_configs
        self.current_train_phase = 0
        self.loss_module = LossRegistry.get_class(self.loss_configs.type)(
            loss_configs=self.loss_configs, dataset_configs=dataset_configs
        )
        self.objective = Objective(dataset_configs=self.dataset_configs, is_criterion=True)
        self.lr = self.train_configs.lr
        self._define_modules(**kwargs)

        # additional loss terms
        self.grad_loss = None
        self.disparity_loss = None
        if self.loss_configs.additional:
            found = self.loss_configs.find_additional_loss("GradientGuidedLoss")
            if found:
                self.grad_loss = LossRegistry.get_class("GradientGuidedLoss")(
                    loss_configs=self.loss_configs, dataset_configs=self.dataset_configs, weight=found.weight
                )
            found = self.loss_configs.find_additional_loss("DisparityLoss")
            if found:
                self.disparity_loss = LossRegistry.get_class("DisparityLoss")(
                    loss_configs=self.loss_configs, weight=found.weight
                )

        self.example_input_array = torch.rand(
            [2, self.dataset_configs.params.K, self.dataset_configs.params.N]
        ) + 1j * torch.rand([2, self.dataset_configs.params.K, self.dataset_configs.params.N])

        self._registered_loss_hooks = TotalLoss([])

    @abstractmethod
    def _define_modules(self, **kwargs) -> None:
        """It should define the torch Modules to be used in the forward
        propagation.

        :return: None
        """

    def add_loss_hook(self, loss: TotalLoss):
        self._registered_loss_hooks.extend(loss)

    def __init_subclass__(cls, **kwargs):
        """used for registering subclasses in the registry."""
        super().__init_subclass__(**kwargs)
        ModelRegistry.register(cls)

    @abstractmethod
    def forward(self, inp: t.Union[torch.Tensor, DatasetItems], mode: TrainMode, **kwargs) -> ModelOutput:
        """Performs forward propagation. Returns ModelOutput that contains at
        least the estimated waveform.

        :param inp: (t.Union[torch.Tensor, DatasetItems]) the output of the DataLoader or the desired beampattern
                    directly as a Tensor
        :param mode: (TrainMode) whether Train, Validate or Test
        :param kwargs: any other inputs that could be helpful and differs from a model to another
        :return: (ModelOutput) it should at least contain an estimated waveform.
        """

    def configure_optimizers(self):
        """only Adam/ReduceLROnPlateau are supported for now."""
        match self.train_configs.optimizer:
            case OptimizerType.Adam:
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            case _:
                raise NotImplementedError(f"unsupported optimizer type: {self.train_configs.optimizer}")
        if self.train_configs.lr_scheduler:
            match self.train_configs.lr_scheduler[0]:
                case LRSchedulerType.ReduceLROnPlateau:
                    min_lr = float(self.train_configs.lr_scheduler[1].min_lr)
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": ReduceLROnPlateau(optimizer=optimizer, min_lr=min_lr),
                            "monitor": "ValidationError(dB)",
                        },
                    }
                case _:
                    raise NotImplementedError(f"Unsupported Scheduler: {self.train_configs.lr_scheduler}")
        else:
            return optimizer

    def on_train_epoch_start(self) -> None:
        """Called at the start of every epoch checks if we use phases in
        training and adjust model appropriately."""
        if self.train_configs.phases:
            if self.current_epoch == 0:
                warnings.warn("updating freeze/unfreeze/lr statuses")
            if (
                len(self.train_configs.phases) > self.current_train_phase + 1
                and self.train_configs.phases[self.current_train_phase + 1].epoch == self.current_epoch
            ):
                self.current_train_phase += 1
                warnings.warn("updating freeze/unfreeze/lr statuses")
            phase_configs = self.train_configs.phases[self.current_train_phase]
            if phase_configs.lr:
                self.lr = phase_configs.lr
            match phase_configs.freeze:
                case TrainPhaseOption.All:
                    self.freeze()
                case TrainPhaseOption.Rest:
                    for name, child in self.named_children():
                        if name in phase_configs.unfreeze:
                            unfreeze_module(child)
                        else:
                            freeze_module(child)
                case TrainPhaseOption.Nothing:
                    self.unfreeze()
                case _:
                    for name, child in self.named_children():
                        if name in phase_configs.freeze:
                            freeze_module(child)
                        else:
                            unfreeze_module(child)

    def training_step(self, train_batch: DatasetItems, batch_idx: int, **kwargs) -> torch.Tensor:
        """calls forward, then calculates loss and logs values."""
        train_model_output = self.forward(inp=train_batch, mode=TrainMode.Train, epoch=self.current_epoch, **kwargs)
        loss = self.loss_module(
            mode=TrainMode.Train,
            input_batch=train_batch,
            model_output=train_model_output,
            epoch=self.current_epoch,
            **kwargs,
        )

        # some models have internal regularization terms that are registered in the registered_loss_hooks
        if self._registered_loss_hooks:
            loss.extend(self._registered_loss_hooks)
            self._registered_loss_hooks = TotalLoss([])

        self.log_dict(loss.logs)
        self.log("loss/train", loss.value.item())
        for callback in self.trainer.callbacks:
            if hasattr(callback, "on_train_batch_loss_end"):
                callback.on_train_batch_loss_end(
                    trainer=self.trainer,
                    model=self,
                    dataset_output=train_batch,
                    model_output=train_model_output,
                    loss_output=loss,
                    batch_idx=batch_idx,
                    unused=0,
                )
        return loss.value

    def validation_step(self, val_batch: DatasetItems, batch_idx: int, **kwargs) -> t.Optional[TotalLoss]:
        """calls forward, then calculates loss and logs values."""
        with torch.no_grad():
            val_model_output = self.forward(inp=val_batch, mode=TrainMode.Validate, epoch=self.current_epoch, **kwargs)
            loss = self.loss_module(
                mode=TrainMode.Validate,
                input_batch=val_batch,
                model_output=val_model_output,
                epoch=self.current_epoch,
                **kwargs,
            )
            db_error = (
                self.objective.estimate_db_diff(
                    x=val_model_output.estimated_waveforms.unsqueeze(dim=1),
                    desired=val_batch.desired_beampatterns,
                )
                .detach()
                .cpu()
            )

            # some models have internal regularization terms that are registered in the registered_loss_hooks
            if self._registered_loss_hooks:
                loss.extend(self._registered_loss_hooks)
                self._registered_loss_hooks = TotalLoss([])

            self.log_dict(loss.logs)
            self.log("loss/validation", loss.value.item())
            self.log("ValidationError(dB)", db_error)
            for callback in self.trainer.callbacks:
                if hasattr(callback, "on_validation_batch_loss_end"):
                    callback.on_validation_batch_loss_end(
                        trainer=self.trainer,
                        model=self,
                        dataset_output=val_batch,
                        model_output=val_model_output,
                        loss_output=loss,
                        batch_idx=batch_idx,
                        unused=0,
                    )
            return loss.value

    def test_step(self, test_batch: DatasetItems, batch_idx: int, **kwargs) -> torch.Tensor:
        """calls forward, then calculates loss and logs values."""
        with torch.no_grad():
            test_model_output = self.forward(inp=test_batch, mode=TrainMode.Test, epoch=self.current_epoch, **kwargs)
            loss = self.loss_module(
                mode=TrainMode.Test,
                input_batch=test_batch,
                model_output=test_model_output,
                epoch=self.current_epoch,
                **kwargs,
            )
            db_error = (
                self.objective.estimate_db_diff(
                    x=test_model_output.estimated_waveforms.unsqueeze(dim=1),
                    desired=test_batch.desired_beampatterns,
                )
                .detach()
                .cpu()
            )

            # some models have internal regularization terms that are registered in the registered_loss_hooks
            if self._registered_loss_hooks:
                loss.extend(self._registered_loss_hooks)
                self._registered_loss_hooks = TotalLoss([])

            self.log_dict(loss.logs)
            self.log("loss/test", loss.value.item())
            self.log("TestError(dB)", db_error)
            for callback in self.trainer.callbacks:
                if hasattr(callback, "on_test_batch_loss_end"):
                    callback.on_test_batch_loss_end(
                        trainer=self.trainer,
                        model=self,
                        dataset_output=test_batch,
                        model_output=test_model_output,
                        loss_output=loss,
                        batch_idx=batch_idx,
                        unused=0,
                    )
            return test_model_output.estimated_waveforms

    def on_validation_epoch_end(self) -> None:
        self.log_dict({"epoch/Num-of-tensors_val": get_num_of_tensors(), "epoch/Cpu-mem-usg_val": get_cpu_mem()})

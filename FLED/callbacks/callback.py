"""Base class used to build new callbacks."""

import typing as t

import torch
import trainer as tr
from callbacks import CallbackRegistry
from models import base_model
from pytorch_lightning import Callback as _Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import Optimizer
from utils.item_classes import DatasetItems, ModelOutput, TotalLoss


class Callback(_Callback):
    """Abstract base class used to build new callbacks.

    Subclass this class and override any of the relevant hooks
    """

    def __init_subclass__(cls, **kwargs):
        """used for registering subclasses in the registry."""
        super().__init_subclass__(**kwargs)
        CallbackRegistry.register(cls)

    def setup(self, trainer: "tr.Trainer", model: "base_model.Model", stage: t.Optional[str] = None) -> None:
        """Called when fit, validate, test, predict, or tune begins."""

    def teardown(self, trainer: "tr.Trainer", model: "base_model.Model", stage: t.Optional[str] = None) -> None:
        """Called when fit, validate, test, predict, or tune ends."""

    def on_fit_start(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when fit begins."""

    def on_fit_end(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when fit ends."""

    def on_sanity_check_start(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the validation sanity check starts."""

    def on_sanity_check_end(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the validation sanity check ends."""

    def on_train_batch_start(
        self,
        trainer: "tr.Trainer",
        model: "base_model.Model",
        batch: t.Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        """Called when the train batch begins."""

    def on_train_batch_end(
        self,
        trainer: "tr.Trainer",
        model: "base_model.Model",
        outputs: STEP_OUTPUT,
        batch: t.Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        """Called when the train batch ends."""

    def on_train_epoch_start(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the train epoch begins."""

    def on_train_epoch_end(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, either:

        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module OR
        2. Cache data across train batch hooks inside the callback implementation to post-process in this hook.
        """

    def on_validation_epoch_start(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the val epoch begins."""

    def on_validation_epoch_end(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the val epoch ends."""

    def on_test_epoch_start(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the test epoch begins."""

    def on_test_epoch_end(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the test epoch ends."""

    def on_predict_epoch_start(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the predict epoch begins."""

    def on_predict_epoch_end(self, trainer: "tr.Trainer", model: "base_model.Model", outputs: list[t.Any]) -> None:
        """Called when the predict epoch ends."""

    def on_validation_batch_start(
        self, trainer: "tr.Trainer", model: "base_model.Model", batch: t.Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the validation batch begins."""

    def on_validation_batch_end(
        self,
        trainer: "tr.Trainer",
        model: "base_model.Model",
        outputs: t.Optional[STEP_OUTPUT],
        batch: t.Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends."""

    def on_test_batch_start(
        self, trainer: "tr.Trainer", model: "base_model.Model", batch: t.Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the test batch begins."""

    def on_test_batch_end(
        self,
        trainer: "tr.Trainer",
        model: "base_model.Model",
        outputs: t.Optional[STEP_OUTPUT],
        batch: t.Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends."""

    def on_predict_batch_start(
        self, trainer: "tr.Trainer", model: "base_model.Model", batch: t.Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the predict batch begins."""

    def on_predict_batch_end(
        self,
        trainer: "tr.Trainer",
        model: "base_model.Model",
        outputs: t.Any,
        batch: t.Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the predict batch ends."""

    def on_train_start(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the train begins."""

    def on_train_end(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the train ends."""

    def on_validation_start(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the validation loop begins."""

    def on_validation_end(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the validation loop ends."""

    def on_test_start(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the test begins."""

    def on_test_end(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the test ends."""

    def on_predict_start(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when the predict begins."""

    def on_predict_end(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called when predict ends."""

    def on_exception(self, trainer: "tr.Trainer", model: "base_model.Model", exception: BaseException) -> None:
        """Called when any trainer execution is interrupted by an exception."""

    def state_dict(self) -> dict[str, t.Any]:
        """Called when saving a checkpoint, implement to generate callback's
        ``state_dict``.

        Returns:
            A dictionary containing callback state.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, t.Any]) -> None:
        """Called when loading a checkpoint, implement to reload callback state
        given callback's ``state_dict``.

        Args:
            state_dict: the callback state returned by ``state_dict``.
        """
        pass

    def on_before_backward(self, trainer: "tr.Trainer", model: "base_model.Model", loss: torch.Tensor) -> None:
        """Called before ``loss.backward()``."""

    def on_after_backward(self, trainer: "tr.Trainer", model: "base_model.Model") -> None:
        """Called after ``loss.backward()`` and before optimizers are
        stepped."""

    def on_before_optimizer_step(
        self, trainer: "tr.Trainer", model: "base_model.Model", optimizer: Optimizer, opt_idx: int
    ) -> None:
        """Called before ``optimizer.step()``."""

    def on_before_zero_grad(self, trainer: "tr.Trainer", model: "base_model.Model", optimizer: Optimizer) -> None:
        """Called before ``optimizer.zero_grad()``."""

    def on_train_batch_loss_end(
        self,
        trainer: "tr.Trainer",
        model: "base_model.Model",
        dataset_output: DatasetItems,
        model_output: ModelOutput,
        loss_output: TotalLoss,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        """Called just before the train batch about to end."""

    def on_validation_batch_loss_end(
        self,
        trainer: "tr.Trainer",
        model: "base_model.Model",
        dataset_output: DatasetItems,
        model_output: ModelOutput,
        loss_output: TotalLoss,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        """Called just before the validation batch about to end."""

    def on_test_batch_loss_end(
        self,
        trainer: "tr.Trainer",
        model: "base_model.Model",
        dataset_output: DatasetItems,
        model_output: ModelOutput,
        loss_output: TotalLoss,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        """Called just before the test batch about to end."""

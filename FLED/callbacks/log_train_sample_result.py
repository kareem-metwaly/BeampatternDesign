import typing as t

from callbacks.callback import Callback
from losses.objective import Objective
from pytorch_lightning import LightningModule as Model
from pytorch_lightning import Trainer
from utils.item_classes import DatasetItems, ModelOutput, TotalLoss


class LogTrainSampleResult(Callback):
    objective: Objective

    def setup(self, trainer: Trainer, model: Model, stage: t.Optional[str] = None) -> None:
        super().__init__()
        self.objective = Objective(dataset_configs=model.dataset_configs)

    def on_fit_start(self, trainer: Trainer, model: Model) -> None:
        """Called when fit begins."""
        self.objective = Objective(dataset_configs=model.dataset_configs).to(model.device)

    def on_train_batch_loss_end(
        self,
        trainer: Trainer,
        model: Model,
        dataset_output: DatasetItems,
        model_output: ModelOutput,
        loss_output: TotalLoss,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        """Called just before the train batch about to end."""
        metrics = {
            "Objective": self.objective.core_forward(
                x=model_output.estimated_waveforms, desired=dataset_output.desired_beampatterns
            ),
        }
        trainer.logger.log_metrics(metrics)

    def on_validation_batch_loss_end(
        self,
        trainer: Trainer,
        model: Model,
        dataset_output: DatasetItems,
        model_output: ModelOutput,
        loss_output: TotalLoss,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        """Called just before the validation batch about to end."""

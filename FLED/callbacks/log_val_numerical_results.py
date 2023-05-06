"""implementation of logging callback at validation to save the estimated
beampattern and the actual desired one."""

import typing as t

import torch
from callbacks.callback import Callback
from losses.objective import Objective
from models.base_model import Model
from trainer import Trainer
from utils.item_classes import DatasetItems, ModelOutput, TotalLoss


class LogValNumericalResult(Callback):
    """at validation, it saves images of estimated and desired beampatterns."""

    objective: Objective
    # first tensor is the error, second tensor is dB error
    cache: dict[t.Literal["PDR", "Unrolled", "Initial"], list[torch.Tensor]]
    PDR_cache: t.Optional[t.Sequence[torch.Tensor]]
    histogram_log_freq: int

    def setup(self, trainer: Trainer, model: Model, stage: t.Optional[str] = None) -> None:
        """called in the beginning to define attributes that will be used
        later: objective is used to find the estimated beampattern."""
        super().__init__()
        self.objective = Objective(dataset_configs=model.dataset_configs, is_criterion=True)
        self.PDR_cache = None
        self.cache = {
            "PDR": [torch.Tensor(), torch.Tensor()],
            "Unrolled": [torch.Tensor(), torch.Tensor()],
            "Initial": [torch.Tensor(), torch.Tensor()],
        }
        self.histogram_log_freq = 20

    def on_fit_start(self, trainer: Trainer, model: Model) -> None:
        """Called when fit begins."""
        self.objective = self.objective.to(model.device)
        trainer.logger.experiment.add_custom_scalars(
            {
                "Numerical": {
                    "Means": ["Multiline", [f"Means/{name}" for name in self.cache.keys()]],
                    "Stds": ["Multiline", [f"Stds/{name}" for name in self.cache.keys()]],
                    "dBError": ["Multiline", [f"dBError/{name}" for name in self.cache.keys()]],
                }
            }
        )

    def on_test_start(self, trainer: Trainer, model: Model) -> None:
        self.on_fit_start(trainer, model)

    def on_test_batch_loss_end(
        self,
        trainer: Trainer,
        model: Model,
        dataset_output: DatasetItems,
        model_output: ModelOutput,
        loss_output: TotalLoss,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        self.on_validation_batch_loss_end(trainer, model, dataset_output, model_output, loss_output, batch_idx, unused)

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
        """Called just before the validation batch about to end.

        caches the errors for PDR and unrolling algorithm to have a
        comparison at the end of each epoch
        """
        estimated = model_output.estimated_waveforms.detach()
        initial = model_output.initial.detach()
        optimum = dataset_output.optimum_waveforms.detach()
        if optimum.shape[-1] == 0:
            optimum = None
        desired = dataset_output.desired_beampatterns.detach()
        self.cache["Unrolled"] = [
            torch.cat(
                (
                    self.cache["Unrolled"][0],
                    self.objective(x=estimated.unsqueeze(dim=1), desired=desired, reduce=False)
                    .mean(dim=[1, 2, 3])
                    .cpu(),
                ),
                dim=0,
            ),
            torch.cat(
                (
                    self.cache["Unrolled"][1],
                    self.objective.estimate_db_diff(x=estimated.unsqueeze(dim=1), desired=desired).cpu(),
                ),
                dim=0,
            ),
        ]

        if model_output.initial is not None:
            self.cache["Initial"] = [
                torch.cat(
                    (
                        self.cache["Initial"][0],
                        self.objective(x=initial, desired=desired, reduce=False).mean(dim=[1, 2, 3]).cpu(),
                    ),
                    dim=0,
                ),
                torch.cat(
                    (
                        self.cache["Initial"][1],
                        self.objective.estimate_db_diff(x=initial, desired=desired).cpu(),
                    ),
                    dim=0,
                ),
            ]

        if self.PDR_cache is None:
            if optimum is None:
                self.cache["PDR"] = [torch.zeros([1]), torch.zeros([1])]
            else:
                self.cache["PDR"] = [
                    torch.cat(
                        (
                            self.cache["PDR"][0],
                            self.objective(x=optimum.unsqueeze(dim=1), desired=desired, reduce=False)
                            .mean(dim=[1, 2, 3])
                            .cpu(),
                        ),
                        dim=0,
                    ),
                    torch.cat(
                        (
                            self.cache["PDR"][1],
                            self.objective.estimate_db_diff(x=optimum.unsqueeze(dim=1), desired=desired).cpu(),
                        ),
                        dim=0,
                    ),
                ]
        del estimated
        del optimum
        del desired

    def on_validation_epoch_end(self, trainer: Trainer, model: Model) -> None:
        """Called when the val epoch ends.

        calculates the histogram / statistics of the whole validation
        values
        """
        if self.PDR_cache is None:
            self.PDR_cache = self.cache["PDR"]

        for name, vals in self.cache.items():
            if (name != "PDR" or trainer.current_epoch == 0) and (trainer.current_epoch % self.histogram_log_freq == 0):
                trainer.logger.experiment.add_histogram(f"Hist/{name}", vals[0], trainer.current_epoch)
                trainer.logger.experiment.add_histogram(f"dBHist/{name}", vals[1], trainer.current_epoch)
            trainer.logger.experiment.add_scalar(f"Means/{name}", vals[0].mean(), trainer.current_epoch)
            trainer.logger.experiment.add_scalar(f"Stds/{name}", vals[0].std(dim=0), trainer.current_epoch)
            trainer.logger.experiment.add_scalar(f"dBError/{name}", vals[1].mean(), trainer.current_epoch)

        # make sure that all tensors are freed as it could be causing memory overflow
        for name in self.cache.keys():
            # don't free PDR as we will use the same values again
            if name != "PDR":
                size = len(self.cache[name])
                for _ in range(len(self.cache[name])):
                    del self.cache[name][0]
                self.cache[name] = [torch.Tensor() for _ in range(size)]

"""implementation of logging callback at validation to save the estimated
beampattern and the actual desired one."""

import typing as t
import warnings

# import h5py
import torch
import torchvision
from callbacks.callback import Callback
from losses.objective import Objective
from matplotlib.cm import hot
from models.base_model import Model
from trainer import Trainer
from utils.item_classes import DatasetItems, ModelOutput, TotalLoss

# from pathlib import Path


class LogValBeampatternResult(Callback):
    """at validation, it saves images of estimated and desired beampatterns."""

    objective: Objective
    log_every: int

    def setup(self, trainer: Trainer, model: Model, stage: t.Optional[str] = None) -> None:
        """called in the beginning to define attributes that will be used
        later: objective is used to find the estimated beampattern."""
        super().__init__()
        self.objective = Objective(dataset_configs=model.dataset_configs, is_criterion=True)
        self.log_every = 100

    def on_fit_start(self, trainer: Trainer, model: Model) -> None:
        """Called when fit begins."""
        self.objective = self.objective.to(model.device)

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

        calculates the generated desired beampatterns
        """
        if batch_idx % self.log_every == 0:
            actual = dataset_output.desired_beampatterns.detach().unsqueeze(dim=1).abs()  # size B x 1 x K x N
            estimated = model_output.estimated_waveforms.detach().unsqueeze(dim=1)
            optimum = dataset_output.optimum_waveforms.detach().unsqueeze(dim=1)
            estimated = self.objective.estimated_beampattern(x=estimated, normalize=True)  # size B x 1 x K x N

            if not optimum.shape[-1]:
                optimum = None
                images = torch.cat((estimated, actual), dim=2)  # size B x 1 x 3.K x N
            else:
                optimum = self.objective.estimated_beampattern(x=optimum, normalize=True)  # size B x 1 x K x N
                images = torch.cat((estimated, optimum, actual), dim=2)  # size B x 1 x 3.K x N

            # batch_size = trainer.configs.batch_size
            # with h5py.File(Path(trainer.log_dir).joinpath("validation_results.h5"), "a") as file:
            #     desired_beampattern = actual.detach().squeeze(dim=1).abs().cpu().numpy()
            #     estimated_beampattern = estimated.detach().squeeze(dim=1).abs().cpu().numpy()
            #     shape = desired_beampattern.shape[1:]
            #     for index in range(batch_size):
            #         idx = batch_idx * batch_size + index
            #         file.create_dataset(f"{idx}_desired", shape, data=desired_beampattern[index])
            #         file.create_dataset(f"{idx}_estimated", shape, data=estimated_beampattern[index])

            # convert to colored scale
            images = torch.Tensor(hot(images.squeeze(dim=1).cpu().numpy())).permute((0, 3, 1, 2))[:, :3]
            images = torchvision.utils.make_grid(images)
            if images.max() > 1:
                warnings.warn("some values exceed 1")
            if images.min() < 0:
                warnings.warn("some values are negative")

            trainer.logger.experiment.add_image(f"estimate_actual_{batch_idx}", images, trainer.current_epoch)
            del images

"""implementation of logging callback at validation to save the estimated
beampattern and the actual desired one for the pre-specified scenarios in the
dataset."""

import typing as t
from pathlib import Path

import h5py
import torch
from callbacks.callback import Callback
from datasets.scenarios import ScenariosBeamPattern
from losses.objective import Objective
from matplotlib.cm import hot
from models.base_model import Model
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer import Trainer
from utils.item_classes import DatasetItems


class LogScenarioResult(Callback):
    """at validation, it saves images of estimated and desired beampatterns."""

    scenarios: ScenariosBeamPattern
    objective: Objective
    # first tensor is the error, second tensor is dB error
    cache: dict[t.Literal["PDR", "Unrolled", "Initial"], list[torch.Tensor]]
    PDR_cache: t.Optional[t.Sequence[torch.Tensor]]
    histogram_log_freq: int

    def setup(self, trainer: Trainer, model: Model, stage: t.Optional[str] = None) -> None:
        """called in the beginning to define attributes that will be used
        later: scenarios is used to generate pre-specified beampattern."""
        super().__init__()
        self.objective = Objective(dataset_configs=model.dataset_configs, is_criterion=True)
        self.scenarios = DataLoader(
            ScenariosBeamPattern(trainer._datamodule.dataset_configs), collate_fn=DatasetItems.collate
        )
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
                    "ScenarioMeans": ["Multiline", [f"ScenarioMeans/{name}" for name in self.cache.keys()]],
                    "ScenarioStds": ["Multiline", [f"ScenarioStds/{name}" for name in self.cache.keys()]],
                    "ScenariodBError": ["Multiline", [f"ScenariodBError/{name}" for name in self.cache.keys()]],
                }
            }
        )
        path = Path(trainer.logger.experiment.get_logdir())
        path.mkdir(parents=True, exist_ok=True)
        with h5py.File(path.joinpath("scenarios.h5"), "w") as file:
            for idx, sample in enumerate(tqdm(self.scenarios, desc="dumping scenarios", position=-2, mininterval=100)):
                desired_beampattern = sample.desired_beampatterns.detach().squeeze().cpu().numpy()
                file.create_dataset(str(idx), desired_beampattern.shape, data=desired_beampattern)
        print("finished dumping scenarios")

    def on_test_start(self, trainer: Trainer, model: Model) -> None:
        self.on_fit_start(trainer, model)

    def on_test_end(self, trainer: Trainer, model: Model) -> None:
        self.on_validation_epoch_end(trainer, model)

    def on_validation_epoch_end(self, trainer: Trainer, model: Model) -> None:
        """Called when the val epoch ends.

        Loops through scenarios and store their results calculates the
        histogram / statistics of the whole validation values
        """

        with torch.no_grad():
            for idx, dataset_output in enumerate(self.scenarios):
                # ToPILImage()(dataset_output.desired_beampatterns.abs()).show()
                # plt.imshow(dataset_output.desired_beampatterns.abs()[0].numpy())
                # plt.show()
                dataset_output = dataset_output.to(model.device)
                model_output = model(dataset_output, epoch=trainer.current_epoch, compute_loss=False)
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

                trainer.logger.experiment.add_scalar(f"dB/{idx}", self.cache["Unrolled"][1][-1], trainer.current_epoch)
                estimated = self.objective.estimated_beampattern(
                    x=estimated.unsqueeze(dim=1), normalize=True
                )  # size B x 1 x K x N)
                initial = self.objective.estimated_beampattern(
                    x=initial[:, 0].unsqueeze(dim=1), normalize=True
                ).squeeze()  # size B x 1 x K x N)
                actual = dataset_output.desired_beampatterns.detach().unsqueeze(dim=1)
                images = torch.cat((estimated, actual), dim=3).cpu().abs()  # size B x 1 x 3.K x N
                # optimum = self.objective.estimated_beampattern(x=optimum, normalize=True)  # size B x 1 x K x N
                # images = torch.cat((estimated, optimum, actual), dim=2)  # size B x 1 x 3.K x N
                data = hot(images.squeeze().abs().cpu().numpy())

                with h5py.File(Path(trainer.log_dir).joinpath("scenario_results.h5"), "a") as file:
                    desired_beampattern = actual.detach().squeeze().abs().cpu().numpy()
                    estimated_beampattern = estimated.detach().squeeze().cpu().numpy()
                    initial_beampattern = initial.detach().squeeze().cpu().numpy()
                    file.create_dataset(f"{idx}_desired", desired_beampattern.shape, data=desired_beampattern)
                    file.create_dataset(f"{idx}_estimated", estimated_beampattern.shape, data=estimated_beampattern)
                    file.create_dataset(f"{idx}_initial", initial_beampattern.shape, data=estimated_beampattern)

                # convert to colored scale
                images = torch.Tensor(data).permute(2, 1, 0)[:3]
                trainer.logger.experiment.add_image(f"scenario_{idx}", images, trainer.current_epoch)

        if self.PDR_cache is None:
            self.PDR_cache = self.cache["PDR"]

        for name, vals in self.cache.items():
            if (name != "PDR" or trainer.current_epoch == 0) and (trainer.current_epoch % self.histogram_log_freq == 0):
                trainer.logger.experiment.add_histogram(f"ScenarioHist/{name}", vals[0], trainer.current_epoch)
                trainer.logger.experiment.add_histogram(f"ScenariodBHist/{name}", vals[1], trainer.current_epoch)
            trainer.logger.experiment.add_scalar(f"ScenarioMeans/{name}", vals[0].mean(), trainer.current_epoch)
            trainer.logger.experiment.add_scalar(f"ScenarioStds/{name}", vals[0].std(dim=0), trainer.current_epoch)
            trainer.logger.experiment.add_scalar(f"ScenariodBError/{name}", vals[1].mean(), trainer.current_epoch)

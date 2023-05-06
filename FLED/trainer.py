"""Trainer class, inheriting from pytorch_lightning class and tailored to our
problem formulation."""
import threading  # noqa F401
import typing as t

import callbacks.callback as cb
import pytorch_lightning as pl
from datasets.base_dataset import BaseDataLoader
from models.base_model import Model
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.transforms.functional import to_tensor
from utils.config_classes import TrainConfig
from utils.etc import fig2img, find_latest_checkpoint


class Trainer(pl.Trainer):
    """Custom trainer inheriting from PytorchLightning.

    Main reason to use it, is to allow usage of configurations and
    passing them easily It can be further used to develop a custom
    training steps / logging.
    """

    configs: TrainConfig
    callbacks: t.Sequence["cb.Callback"]
    logger: TensorBoardLogger
    _datamodule: BaseDataLoader
    _model: Model  # we use underscore as pytorch-lighting trainer has the property model

    def __init__(
        self,
        configs: TrainConfig,
        model: Model,
        datamodule: BaseDataLoader,
        logger: TensorBoardLogger,
        callbacks: list["cb.Callback"] = None,
        **kwargs,
    ):
        """stores the configs and invokes the trainer of pytorch lightning. It
        also calls __post_init__.

        :param configs:
        :param model:
        :param datamodule:
        :param callbacks: (Callback)
        :param logger:
        :param kwargs:
        """
        self.configs = configs
        self._datamodule = datamodule
        self._model = model
        assert isinstance(logger, TensorBoardLogger), f"we must have the first logger as tensorboard, given {logger}"
        if callbacks is None:
            callbacks = [pl.callbacks.LearningRateMonitor()]
        else:
            callbacks.append(pl.callbacks.LearningRateMonitor())
        super().__init__(
            max_epochs=configs.max_epochs,
            auto_lr_find=configs.auto_lr_find,
            logger=logger,
            callbacks=callbacks,
            default_root_dir=configs.checkpoints_dir,
            precision=configs.precision,
            gpus=configs.gpus,
            log_every_n_steps=configs.log_every_n_steps,
            gradient_clip_val=configs.gradient_clip_val,
            val_check_interval=configs.val_check_interval,
            # profiler="pytorch",
            **kwargs,
        )
        self.logger.log_hyperparams(
            params=dict(
                [
                    *[(f"train/{k}", v) for k, v in configs],
                    *[(f"model/{k}", v) for k, v in self._model.model_configs],
                    *[(f"dataset/{k}", v) for k, v in self._datamodule.dataset_configs],
                    *[(f"loss/{k}", v) for k, v in self._model.loss_configs],
                ]
            ),
            metrics={"ValidationError(dB)": 0},
        )
        self.__post_init__()

    def __post_init__(self):
        """for now, it attempts to know the best lr if it is set in configs.

        :return: None
        """
        self._model.hparams.lr = self.configs.lr
        self.logger.log_graph(
            self._model,
            input_array=self._datamodule.train_dataloader().dataset[0].desired_beampattern.unsqueeze(0),
        )
        # threading.Thread(target=self._datamodule.dump_val_data, args=[self.logger.log_dir]).start()
        if self.configs.auto_lr_find:
            lr_finder = self.tuner.lr_find(
                model=self._model,
                datamodule=self._datamodule,
            )
            # plotting the results
            fig = lr_finder.plot(suggest=True)
            fig.show()
            self.logger.experiment.add_image("lr_finder_results", to_tensor(fig2img(fig)))
            new_lr = lr_finder.suggestion()
            self.configs.lr = new_lr

    def fit(self, **kwargs) -> None:
        """performs fitting as Pytorch Lightning, giving it all the necessary
        args."""
        super().fit(
            self._model,
            datamodule=self._datamodule,
            ckpt_path=find_latest_checkpoint(self.configs.checkpoints_dir),
        )

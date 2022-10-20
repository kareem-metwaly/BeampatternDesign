"""This file is used to train the network.

It takes some arguments through argparse that specifies which model,
loss and dataset to load.
"""
import typing as t
from argparse import ArgumentParser

from callbacks import CallbackRegistry
from callbacks.callback import Callback
from datasets.base_dataset import BaseDataLoader, DataLoaderRegistry
from models import ModelRegistry
from models.base_model import Model
from pydantic import FilePath
from pytorch_lightning import loggers
from trainer import Trainer
from utils.base_classes import ConfigType
from utils.config_classes import DatasetConfig, LossConfig, ModelConfig, TrainConfig
from utils.ProjectDirectory import ProjectDirectory


def train(
    dataset_configs: FilePath,
    train_configs: FilePath,
    model_configs: FilePath,
    loss_configs: FilePath,
    model: t.Type[Model],
    datamodule: t.Type[BaseDataLoader],
    trainer: t.Type[Trainer],
    callbacks: t.Optional[t.Union[t.Type[Callback], list[t.Type[Callback]]]],
) -> None:

    """the main training code: invokes dataloader, model, losses and performs
    training."""
    dataset_configs = DatasetConfig.parse_yaml(dataset_configs)
    model_configs = ModelConfig.parse_yaml(model_configs)
    loss_configs = LossConfig.parse_yaml(loss_configs)
    train_configs = TrainConfig.parse_yaml(train_configs)

    dataloader_module = datamodule(train_configs=train_configs, dataset_configs=dataset_configs)
    model = model(
        model_configs=model_configs,
        train_configs=train_configs,
        loss_configs=loss_configs,
        dataset_configs=dataset_configs,
    )
    if not isinstance(callbacks, list):
        callbacks = [callbacks]
    callbacks = [callback() for callback in callbacks if callback is not None]

    trainer = trainer(
        configs=train_configs,
        model=model,
        datamodule=dataloader_module,
        callbacks=callbacks,
        logger=loggers.TensorBoardLogger(
            save_dir=str(ProjectDirectory.logs_path("TensorBoard")),
            log_graph=True,
            default_hp_metric=False,
        ),
    )
    trainer.fit()


if __name__ == "__main__":
    args = ArgumentParser()

    # define modules arguments
    args.add_argument(
        "--model",
        type=str,
        choices=ModelRegistry.classes(),
        default=next(iter(ModelRegistry.classes())),
        help="Which model to use",
    )
    args.add_argument(
        "--datamodule",
        type=str,
        choices=DataLoaderRegistry.classes(),
        default=next(iter(DataLoaderRegistry.classes())),
        help="Which dataloader to use",
    )

    # define config arguments
    args.add_argument(
        "--model-config",
        type=str,
        choices=ProjectDirectory.list_configs(ConfigType.Models),
        default=ProjectDirectory.list_configs(ConfigType.Models)[0],
        help="Which model configs to use",
    )
    args.add_argument(
        "--train-config",
        type=str,
        choices=ProjectDirectory.list_configs(ConfigType.Train),
        default=ProjectDirectory.list_configs(ConfigType.Train)[0],
        help="Which train configs to use",
    )
    args.add_argument(
        "--dataset-config",
        type=str,
        choices=ProjectDirectory.list_configs(ConfigType.Datasets),
        default=ProjectDirectory.list_configs(ConfigType.Datasets)[0],
        help="Which dataset configs to use",
    )
    args.add_argument(
        "--loss-config",
        type=str,
        choices=ProjectDirectory.list_configs(ConfigType.Losses),
        default=ProjectDirectory.list_configs(ConfigType.Losses)[0],
        help="Which loss configs to use",
    )
    args.add_argument(
        "--callbacks",
        type=str,
        nargs="*",
        choices=CallbackRegistry.classes(),
        default=None,
        help="Which callbacks to use",
    )

    args = args.parse_args()

    train(
        model_configs=ProjectDirectory.configs_path(ConfigType.Models, args.model_config),
        dataset_configs=ProjectDirectory.configs_path(ConfigType.Datasets, args.dataset_config),
        train_configs=ProjectDirectory.configs_path(ConfigType.Train, args.train_config),
        loss_configs=ProjectDirectory.configs_path(ConfigType.Losses, args.loss_config),
        model=ModelRegistry.get_class(args.model),
        trainer=Trainer,
        datamodule=DataLoaderRegistry.get_class(args.datamodule),
        callbacks=[CallbackRegistry.get_class(callback) for callback in args.callbacks] if args.callbacks else None,
    )

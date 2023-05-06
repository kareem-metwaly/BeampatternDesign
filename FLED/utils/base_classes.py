"""Contains all the base classes that are used in the code."""

import typing as t
from abc import ABCMeta, abstractmethod
from enum import Enum
from pathlib import Path

import torch
from pydantic import BaseModel
from pymatreader import read_mat
from utils.etc import h5_load
from utils.types import FilePath, Tensor1D, Tensor2D
from utils.yaml import YamlLoader

# Some Base Classes ##################


class PreBaseModel(BaseModel):
    """performs some checks before initialization of the object."""

    _pre_model: bool = True

    def __init__(self, **kwargs: t.Any) -> None:
        """calls __pre_init__ before initialization.

        :param kwargs:
        """
        self.__pre_init__(**kwargs)
        super().__init__(**kwargs)

    @abstractmethod
    def __pre_init__(self, **kwargs: t.Any) -> None:
        """pre initialization method to be implemented."""


class PostBaseModel(BaseModel):
    """performs some checks after initialization of the object."""

    _post_model: bool = True

    def __init__(self, **kwargs: t.Any) -> None:
        """calls __post_init__ after initialization.

        :param kwargs:
        """
        super().__init__(**kwargs)
        self.__post_init__(**kwargs)

    @abstractmethod
    def __post_init__(self, **kwargs: t.Any) -> None:
        """performs some checks after initialization.

        For now, it loops on variables and whenever it finds a Path, it
        converts it to absolute path
        """
        for var_name in self.__dict__.keys():
            variable = getattr(self, var_name)
            if isinstance(variable, Path):
                setattr(self, var_name, variable.absolute())
                if variable.is_dir():
                    variable.mkdir(parents=True, exist_ok=True)


class PrePostBaseModel(PreBaseModel, PostBaseModel, metaclass=ABCMeta):
    """performs pre- and post-checks."""

    def __init__(self, **kwargs: t.Any) -> None:
        """calls pre- and post-methods.

        :param kwargs:
        """
        self.__pre_init__(**kwargs)
        super().__init__(**kwargs)
        self.__post_init__(**kwargs)


class ArbitraryBaseModel(PostBaseModel, metaclass=ABCMeta):
    """This class must implement a __post__init__ method for arbitrary types.

    Validation has to be done manually here.
    """

    _arbitrary_types: bool = True

    class Config:
        """to ignore unknown data types in validation."""

        arbitrary_types_allowed = True


class ParsingBaseModel(BaseModel, metaclass=ABCMeta):
    """parses a configuration from yaml, h5 or matlab file."""

    _parsing_model: bool = True

    @classmethod
    def parse_yaml(cls: t.Type["ParsingBaseModel"], path: FilePath) -> "ParsingBaseModel":
        """parse from yaml file.

        :param path:
        :return:
        """
        return cls(**YamlLoader.read(path=path))

    @classmethod
    def parse_h5(cls: t.Type["ParsingBaseModel"], path: FilePath) -> "ParsingBaseModel":
        """parses from h5 file.

        :param path:
        :return:
        """
        return cls(**h5_load(path=path, dataset=cls.__name__))

    @classmethod
    def parse_mat(cls: t.Type["ParsingBaseModel"], path: FilePath) -> "ParsingBaseModel":
        """parses form matlab file.

        :param path:
        :return:
        """
        return cls(**read_mat(filename=path, variable_names=cls.__name__).get(cls.__name__))


# Criterion ranks first arg (torch.Tensor) based on a desired second arg (torch.Tensor) and return how close it is
Criterion = t.NewType(
    "Criterion", t.Callable[[Tensor2D, t.Union[Tensor2D, Tensor1D], t.Optional[bool]], torch.FloatTensor]
)


class ConfigType(Enum):
    """used to define different configuration directories we have."""

    Train = "train"
    Models = "models"
    Losses = "losses"
    Datasets = "datasets"
    Undefined = ""


class TrainMode(Enum):
    """Used to indicate which status of training are we in right now."""

    Train = "train"
    Validate = "validate"
    Test = "test"
    Unspecified = "unspecified"

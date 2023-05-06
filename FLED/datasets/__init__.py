"""defines the DatasetRegistry which is used to invoke any new dataset class we
create."""
import typing as t
from enum import Enum

from utils.etc import import_all_submodules
from utils.registry import Registry


class DatasetRegistry(Registry):
    """DatasetRegistry stores all dataset classes and stores a way to retrieve
    them by name."""

    get_class: t.Callable[[t.Union[str, Enum]], t.Type["BaseDataset"]]  # NOQA: F821


class DataLoaderRegistry(Registry):
    """stores all dataset loaders classes."""

    get_class: t.Callable[[t.Union[str, Enum]], t.Type["BaseDataLoader"]]  # NOQA: F821


import_all_submodules(__file__)

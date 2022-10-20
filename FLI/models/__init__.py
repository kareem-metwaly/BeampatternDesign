"""defines the ModelRegistry which is used to invoke any new model class we
create."""
import typing as t
from enum import Enum

from utils.etc import import_all_submodules
from utils.registry import Registry


class ModelRegistry(Registry):
    """ModelRegistry stores all loss classes and stores a way to retrieve them
    by name."""

    get_class: t.Callable[[t.Union[str, Enum]], t.Type["Model"]]  # NOQA: F821


import_all_submodules(__file__)

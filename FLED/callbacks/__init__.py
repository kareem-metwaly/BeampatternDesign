"""defines the CallbackRegistry which is used to invoke any new callback class we
create."""
import typing as t
from enum import Enum

from utils.etc import import_all_submodules
from utils.registry import Registry


class CallbackRegistry(Registry):
    """CallbackRegistry stores all callback classes and stores a way to retrieve them
    by name."""

    get_class: t.Callable[[t.Union[str, Enum]], t.Type["Callback"]]  # NOQA: F821


import_all_submodules(__file__)

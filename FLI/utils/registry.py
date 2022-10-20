"""defines a basic class that is used to store children of a certain class."""
import typing as t
from enum import Enum


class RegistryType(type):
    """resets the value of the registry to empty."""

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls._registry = {}


class Registry(metaclass=RegistryType):
    """A generic registry used to make sure that we create different models,
    losses, ..

    etc and store them appropriately Later we can fetch any class we
    want through `get_class` method.
    """

    _registry: t.Dict[str, t.Type]

    @classmethod
    def register(cls, registered_class: t.Type):
        """stores new class to the registry."""
        if registered_class.__name__ in cls._registry:
            raise ValueError(f"The provided class is already registered, given: {registered_class.__name__}")
        cls._registry[registered_class.__name__] = registered_class

    @classmethod
    def get_class(cls, name: t.Union[str, Enum]) -> t.Type:
        """retrieves a pre-stored class from the registry."""
        if name not in cls._registry:
            raise ValueError(
                f"The provided class is not registered, given: {name} "
                f"while registered values are: [{', '.join(cls._registry.keys())}]"
            )
        return cls._registry[name]

    @classmethod
    def classes(cls) -> t.Iterable[str]:
        """lists all available classes names."""
        return cls._registry.keys()

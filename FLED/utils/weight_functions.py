"""implementations of different weight functions for loss terms."""
import typing as t

from utils.config_classes import LossWeightConfig


class WeightRegistry:
    """stores all weight methods can be invoked by name."""

    @staticmethod
    def fraction(args: tuple[float, float]):
        """an inverse function args / epoch."""
        numerator, exponent = args
        numerator = float(numerator)
        exponent = float(exponent)

        def fraction_calc(epoch: int):
            return numerator / (epoch + 1) ** exponent  # since epoch starts with 1

        return fraction_calc

    @classmethod
    def get_method(cls, config: LossWeightConfig) -> t.Callable:
        """retrieves one of the methods defined in this class and defines its
        args.

        :param config:
        :return:
        """
        return cls.__dict__[config.name](args=config.args)

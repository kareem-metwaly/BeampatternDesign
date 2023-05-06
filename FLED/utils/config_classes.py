"""Contains configuration classes."""

import typing as t
import warnings
from enum import Enum

from pydantic import BaseModel
from utils.base_classes import ArbitraryBaseModel, ParsingBaseModel, PostBaseModel
from utils.types import DirectoryPath

# Config Classes ##################


# Dataset configurations ##########


class Parameters(ParsingBaseModel, PostBaseModel):
    """parameters of the physical problem."""

    N: int
    M: int
    K: int
    fc: float
    B: float
    Ts: t.Optional[float]

    def __post_init__(self, **kwargs: t.Any) -> None:
        super().__post_init__(**kwargs)
        if not self.Ts:
            self.Ts = 1 / self.B


class DatasetConfig(ParsingBaseModel, PostBaseModel):
    """used with the dataset module to load data and retrieve samples for
    training/testing."""

    type: str  # the type of the dataset
    data_path: t.Optional[DirectoryPath] = None
    max_length: t.Optional[int] = None
    perform_sanity: bool = False
    params: t.Optional[Parameters] = None

    def __post_init__(self, **kwargs: t.Any) -> None:
        """assert all values are reasonable.

        :param kwargs:
        :return:
        """
        super().__post_init__(**kwargs)

        if not self.params:
            warnings.warn(
                f"Loading the parameters of the dataset from the provided HDF5 file\n"
                f"{self.data_path.joinpath('parameters.mat')}",
                category=ImportWarning,
            )
            self.params = Parameters.parse_mat(self.data_path.joinpath("parameters.mat"))


# Modules configurations ##########


class ActivationType(Enum):
    """Possible supported activation functions for modules."""

    ReLU = "relu"
    Sigmoid = "sigmoid"
    Tanh = "tanh"
    Softmax = "softmax"
    Retract = "retract"  # implemented to make sure that the output vector satisfy the CMC
    CReLU = "crelu"  # using ReLU(Real) + i ReLU(Imaginary)
    ZReLU = "zrelu"  # using identity only if theta between 0 and pi/2, o.w. o/p is 0
    ModReLU = "mod_relu"  # using CplxModReLU implementation
    AdaModReLU = "adaptive_mod_relu"  # using CplxAdaptiveModReLU implementation
    Norm = "norm"  # normalize the input by dividing by l2 norm
    NoActivation = "none"


class NormalizationType(Enum):
    """Possible supported normalization layers for modules."""

    NaiveNorm = "naive_batchnorm"
    ComplexNorm = "complex_batchnorm"
    NoNormalization = "none"


class PoolType(Enum):
    """possible pooling methods to decrease the size of the input."""

    AvgPool = "avgpool"
    MaxPool = "maxpool"


class UpsampleConfig(BaseModel):
    """Upsampling module configs."""

    type: str
    scale_factor: int


class DownsampleConfig(BaseModel):
    """Downsampling module configs."""

    type: PoolType
    kernel: int
    stride: int
    padding: int


class ScaleConfig(PostBaseModel):
    """Configs for scaling layers; either for 1D conv or 2D conv.

    assuming the output should fall in the range [out_min, out_max], and
    the input in the range of [inp_min, inp_max]
    """

    inp_min: float = 0.0
    inp_max: float = 1.0
    out_min: float = 0.0
    out_max: float = 1.0

    def __post_init__(self, **kwargs: t.Any) -> None:
        """min < max."""
        assert self.inp_min < self.inp_max, f"Inconsistent values for min {self.inp_min} and max {self.inp_max}"
        assert self.out_min < self.out_max, f"Inconsistent values for min {self.out_min} and max {self.out_max}"


class ConvConfig(PostBaseModel):
    """Configs for convolution layers; either for 1D conv or 2D conv."""

    scale_input: t.Optional[ScaleConfig] = None
    n_dims: int
    in_channels: int
    out_channels: t.Optional[int]
    kernel_size: int = 1
    stride: t.Optional[int] = 1
    dilation: t.Optional[int] = 1
    padding: t.Optional[int] = 0
    groups: t.Optional[int] = 1
    bias: t.Optional[bool] = True
    norm: t.Optional[NormalizationType] = NormalizationType.NaiveNorm
    activation: t.Optional[ActivationType] = ActivationType.ReLU
    upsample: t.Optional[UpsampleConfig] = None
    downsample: t.Optional[DownsampleConfig] = None
    scale_output: t.Optional[ScaleConfig] = None

    def __post_init__(self, **kwargs):
        """
        assert all values are reasonable
        :return:
        """
        super().__post_init__(**kwargs)
        assert self.n_dims in {1, 2}, f"we only support 1D and 2D configs, given: {self.n_dims}"
        assert (
            not self.downsample or not self.upsample
        ), "we cannot have upsampling and downsampling in the same conv layer"


class FullyConnectedConfig(ParsingBaseModel):
    """Configs for fully connected layer."""

    scale_input: t.Optional[ScaleConfig] = None
    in_channels: t.Optional[int]
    out_channels: t.Optional[int]
    bias: t.Optional[bool] = True
    norm: t.Optional[NormalizationType] = NormalizationType.NaiveNorm
    activation: t.Optional[ActivationType] = ActivationType.ReLU
    dropout: t.Optional[float] = 0.0
    scale_output: t.Optional[ScaleConfig] = None


class ModelConfig(ParsingBaseModel, PostBaseModel):
    """configurations to build a model used in training."""

    n_expand: int  # the size of expansion (Ne)
    n_prune: int  # the size of prune (Np)
    initial: t.Sequence[FullyConnectedConfig]
    direction_evaluation: t.Sequence[FullyConnectedConfig]
    trainable_prune: bool = False
    prune_params: t.Optional[t.Sequence[FullyConnectedConfig]] = None
    expand: t.Sequence[FullyConnectedConfig]
    n_steps: int
    epsilon: float = 1e-4  # used for numerical stability
    shared_weights: bool = True  # whether to use the same network for every step

    def __post_init__(self, **kwargs: t.Any) -> None:
        """assert all values are reasonable.

        :param kwargs:
        :return:
        """
        super().__post_init__(**kwargs)
        if self.trainable_prune:
            assert isinstance(
                self.prune_params, t.Sequence
            ), f"Since prune is trainable, you have to define a sequence of layers, given: {self.prune_params}"
            for layer in self.prune_params:
                assert isinstance(
                    layer, FullyConnectedConfig
                ), f"Since prune is trainable, layer {layer} has to be Conv2dConfig"
        else:
            assert (
                not self.prune_params
            ), f"since prune is not trainable, you shouldn't define prune_params, given: {self.prune_params}"

        assert (
            self.n_steps >= 0
        ), f"The number of steps for the unrolled algorithm has to be an integer greater than 0, given: {self.n_steps}"
        if self.n_steps == 0:
            warnings.warn("We are not using steps of unrolled algorithm, only initializer is used")
        assert (
            0 <= self.epsilon < 1
        ), f"The epsilon value is used for numerical stability, use a reasonable value, given: {self.epsilon}"


# loss configurations ##################


class LossTermTrueType(Enum):
    Waveform = "waveform"
    Beampattern = "beampattern"
    none = ""


class LossWeightConfig(ParsingBaseModel):
    """Creates a weight configurations."""

    name: str
    args: t.Any


class LossTermConfig(ParsingBaseModel):
    """used in customized loss item with multi-terms."""

    weight: t.Union[
        float, LossWeightConfig
    ]  # the weight given to this loss value, it can be a string for a python function of the variable epoch
    type: str  # name of the loss function to use one from "./losses/" folder
    name: t.Optional[str] = ""  # name of the module which we want to record its output and compare to a gt
    gt: LossTermTrueType = LossTermTrueType.none  # name of the ground truth to compare with
    params: t.Optional[t.Mapping[str, t.Any]] = None  # other parameters related to this loss term


class LossConfig(ParsingBaseModel):
    """configs to build the loss module."""

    type: str
    outputs: t.Optional[list[LossTermConfig]] = None
    additional: t.Optional[
        list[LossTermConfig]
    ] = None  # these are not monitored by the BuildLosses for example, they have to be implemented inside the model

    def find_additional_loss(self, loss_type: str) -> t.Optional[LossTermConfig]:
        found = None
        for loss in self.additional:
            if loss.type == loss_type and not found:
                found = loss
            elif loss.type == type and found:
                raise ValueError(f"Only one {loss_type} should be defined, given: {self.additional}")
        return found


# train configurations ##########


class TrainPhaseOption(Enum):
    """available literal options for training phases."""

    All = "all"
    Rest = "remaining"
    Nothing = "nothing"


class OptimizerType(Enum):
    """available optimizers for training."""

    Adam = "adam"


class LRSchedulerType(Enum):
    """available learning rate schedulers."""

    ReduceLROnPlateau = "reduce on plateau"


class LRSchedulerConfigs(ArbitraryBaseModel, ParsingBaseModel):
    """configs for lr scheduler."""

    min_lr: t.Optional[float] = None

    def __post_init__(self, **kwargs) -> None:
        """assert all values are reasonable.

        :param kwargs:
        :return:
        """
        super().__post_init__(**kwargs)


class TrainPhaseConfig(ParsingBaseModel, PostBaseModel):
    """Training phase configurations it's used to define which modules to train
    at each phase."""

    epoch: int
    freeze: t.Optional[t.Union[list[str], TrainPhaseOption]] = TrainPhaseOption.Nothing
    unfreeze: t.Optional[t.Union[list[str], TrainPhaseOption]] = TrainPhaseOption.All
    lr: t.Optional[float] = None

    def __post_init__(self, **kwargs: t.Any) -> None:
        match self.freeze:
            case TrainPhaseOption.All:
                assert self.unfreeze is TrainPhaseOption.Nothing
            case TrainPhaseOption.Rest:
                assert isinstance(self.unfreeze, list)
            case TrainPhaseOption.Nothing:
                assert self.unfreeze is TrainPhaseOption.All
            case _:
                assert isinstance(self.freeze, list)
                assert self.unfreeze is TrainPhaseOption.Rest


class TrainConfig(ParsingBaseModel, PostBaseModel):
    """Training configurations."""

    auto_lr_find: bool
    max_epochs: int
    lr: float  # a starting learning rate; might change due to lr_scheduler
    gradient_clip_val: float = 0.0
    optimizer: OptimizerType
    lr_scheduler: t.Optional[t.Tuple[LRSchedulerType, LRSchedulerConfigs]] = None

    # related to fetching data
    num_workers: int
    batch_size: int
    precision: int
    train_val_test_split: t.Tuple[float, float, float]
    random_seed: t.Optional[int] = None

    # etc
    gpus: list[int]
    checkpoints_dir: DirectoryPath
    log_every_n_steps: int
    val_check_interval: t.Optional[int] = None

    phases: t.Optional[list[TrainPhaseConfig]] = None

    def __post_init__(self, **kwargs: t.Any) -> None:
        """assert all values are reasonable.

        :param kwargs:
        :return:
        """
        super().__post_init__(**kwargs)
        assert (
            self.log_every_n_steps > 0
        ), f"log_every_n_steps must be a positive number, given {self.log_every_n_steps}"
        assert (sum(self.train_val_test_split) == 1) and all(
            val >= 0 for val in self.train_val_test_split
        ), "The split of training, validation and test sets must be non negative and sum to 1"
        if self.auto_lr_find:
            warnings.warn(
                "The given value of the learning rate may not actually be used, " "automatic lr finding is enabled"
            )

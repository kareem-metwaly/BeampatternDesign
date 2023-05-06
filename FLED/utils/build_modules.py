"""contains all required auxiliary modules and functions to build the core
components of the network."""

import typing as t
import warnings
from collections import OrderedDict

import torch
from torch import Tensor, nn
from utils.activations import ComplexReLU, CplxAdaptiveModReLU, CplxModReLU, ZReLU
from utils.base_classes import Criterion
from utils.complex_functions import complex_l2_norm
from utils.complex_normalization import (
    ComplexBatchNorm1d,
    ComplexBatchNorm2d,
    NaiveComplexBatchNorm1d,
    NaiveComplexBatchNorm2d,
)
from utils.config_classes import (
    ActivationType,
    ConvConfig,
    FullyConnectedConfig,
    NormalizationType,
    PoolType,
    ScaleConfig,
)
from utils.types import ActivationModule, Tensor1D, Tensor2D, TensorScaler

default_epsilon = 1e-5


# This tuple helps in storing intermediate output size to check consistency of connected layers
class BuildOutput(t.NamedTuple):
    """Wrapper of the module that is used during testing."""

    Model: nn.Module
    OutputSize: t.Optional[torch.Size] = None


# Some helpful Modules #############


class ComplexAct(nn.Module):
    """As current activation modules don't support complex numbers, this
    wrapper splits (cartesian or polar) -> process (independently) -> combine
    (return to original input form)."""

    def __init__(self, act: t.Type[ActivationModule], use_phase=False, *args, **kwargs):
        super().__init__()
        if "inplace" in kwargs or hasattr(args, "inplace"):
            warnings.warn("You cannot use inplace with complex activation; turning it off.", category=UserWarning)
            if "inplace" in kwargs:
                kwargs["inplace"] = False
            else:
                args.inplace = False

        self.use_phase = use_phase
        self.act = act(*args, **kwargs)
        if self.use_phase:
            self.forward = self.forward_polar
        else:
            self.forward = self.forward_cartesian

    def __repr__(self):
        return (
            self.act.__repr__().replace(self.act.__class__.__name__, "Complex" + self.act.__class__.__name__)[:-1]
            + f"use_phase={self.use_phase})"
        )

    def forward_polar(self, x: Tensor) -> Tensor:
        """performs forward propagation in case of applying the activation to
        the absolute."""
        return self.act(x.abs()) * (1.0j * x.angle()).exp()

    def forward_cartesian(self, x: Tensor) -> Tensor:
        """performs forward propagation in case of applying activation to the
        real and imaginary parts."""
        return self.act(x.real) + 1.0j * self.act(x.imag)


class Squeeze(nn.Module):
    """Squeeze only removes the arbitrary dims with value 1 given in its
    construction Helpful if you want to connect several layers together that
    won't be compatible without adjusting dims It bypasses any problems while
    shrinking the dims."""

    def __init__(self, dims: t.Sequence[int]):
        """uses dims to squeeze the given input, helpful when we want to
        concatenate layers that may have mismatch in dims.

        :param dims:
        """
        super().__init__()
        self.dims = sorted(dims, reverse=True)

    def forward(self, x: Tensor):
        """for each value of dim attempts to squeeze the input starting from
        the largest dimension and ending with the smallest one.

        :param x:
        :return:
        """
        for dim in self.dims:
            x = x.squeeze(dim=dim)
        return x


class DenseBlock(nn.Module):
    """create a dense block."""

    def __init__(self, configs: t.Sequence[ConvConfig]):
        """Builds the modules.

        :param configs:
        """
        super().__init__()
        modules = []
        for i, c in enumerate(configs):
            modules.append(build_conv(c).Model)

        self.modules = nn.ModuleList(modules)

    def forward(self, x: Tensor) -> Tensor:
        """go through layers.

        :param x:
        :return:
        """
        out = []
        for layer in self.modules:
            x = layer(x)
            out.append(x)
        return torch.cat(out, dim=1)


class Retract(nn.Module):
    """performs retraction step of the PDR algorithm by simply dividing by the
    absolute value (element wise) it uses epsilon for numerical stability."""

    epsilon: float = default_epsilon  # numerical stability parameter

    def __init__(self, epsilon: float = default_epsilon):
        """sets the epsilon value.

        :param epsilon:
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: Tensor1D) -> Tensor1D:
        """performs normalization.

        :param x:
        :return:
        """
        return x / torch.clamp(x.abs(), min=self.epsilon)


class Normalize(nn.Module):
    """performs l2 normalization, it uses epsilon for numerical stability."""

    epsilon: float = default_epsilon  # numerical stability parameter

    def __init__(self, epsilon: float = default_epsilon):
        """sets the epsilon value.

        :param epsilon:
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: Tensor1D) -> Tensor1D:
        """performs normalization.

        :param x:
        :return:
        """
        return x / complex_l2_norm(x, dim=-1, keepdim=True).clamp(min=self.epsilon)


class Scale(nn.Module):
    """assuming the input is in the range of [inp_min, inp_max], it scales the
    output to fall in the range [out_min, out_max]"""

    shift: float
    slope: float

    def __init__(self, config: ScaleConfig):
        """stores the slope, shift values that are used later in forward
        pass."""
        super().__init__()
        inp_range = config.inp_max - config.inp_min
        out_range = config.out_max - config.out_min
        self.slope = out_range / inp_range
        self.shift = config.out_min - config.inp_min * self.slope

    def forward(self, x: Tensor) -> Tensor:
        """scales the output to fall in the range [out_min, out_max]"""
        return x * self.slope + self.shift


class Project(nn.Module):
    """performs projection step (similar to the PDR algorithm)."""

    @staticmethod
    def forward(x: Tensor1D, eta: Tensor1D, beta: TensorScaler) -> Tensor1D:
        """performs projection and move depending on the projected gradient and
        step.

        :param x:
        :param eta:
        :param beta:
        :return:
        """
        projected_eta = eta - (eta.conj() * x).real * x
        return x + beta * projected_eta


class FixedPrune(nn.Module):
    """performs fixed prune; according to the min_criterion it removes input
    tensors with the largest values of min_criterion."""

    output_size: int  # how many to keep
    min_criterion: Criterion  # rank the input based on a desired output (pick min vals for min_criterion)

    def __init__(self, output_size: int, min_criterion: Criterion):
        """sets the size and criterion function.

        :param output_size:
        :param min_criterion:
        """
        super().__init__()
        self.output_size = output_size
        self.min_criterion = min_criterion

    def forward(self, x: Tensor1D, desired: Tensor2D) -> Tensor1D:
        """selects the best values of x that gets as close as possible to the
        desired tensor based on the criterion function.

        :param x:
        :param desired:
        :return:
        """
        _, idx = (
            self.min_criterion(x, desired, reduce=False)
            .sum(dim=[2, 3])
            .unsqueeze(dim=2)
            .topk(k=self.output_size, dim=1, largest=False, sorted=False)
        )
        return x.gather(dim=1, index=idx.repeat([1, 1, x.shape[2]]))


class TrainablePrune(nn.Module):
    """performs prune with trainable parameters; it removes some waveforms
    depending on the ranking generated by the neural network."""

    output_size: int  # how many to keep
    network: nn.Module

    def __init__(self, configs: t.Sequence[FullyConnectedConfig], output_size: int):
        """sets the size and criterion function as well as building the
        network.

        :param configs: (Sequence[FullyConnectedConfig]) to build the network for pruning
        :param output_size: (int) how many to keep after pruning
        """
        super().__init__()
        self.output_size = output_size
        self.network = build_fc_network(configs).Model

    # noinspection PyUnusedLocal
    def forward(self, x: Tensor1D, desired: t.Literal[None] = None) -> Tensor1D:
        """selects the best values of x depending on the top values of the
        network output.

        :param x: (Tensor) the vectors to be pruned
        :param desired: (None) not really used but kept for consistency with the other one
        :return: (Tensor) after pruning and removing far from desired
        """
        _, idx = self.network(x).topk(k=self.output_size, dim=1, largest=False, sorted=False)
        return x[idx]
        # B, _, S = x.shape
        # x = self.network(x.view(B, -1))
        # return x.view(B, self.output_size, S)


class TrainableExpand(nn.Module):
    """performs expansion with trainable parameters; it adds additional
    waveform candidates."""

    output_size: int
    network: nn.Module

    def __init__(self, configs: t.Sequence[FullyConnectedConfig], output_size: int):
        """sets the size and builds the network.

        :param configs: (Sequence[FullyConnectedConfig]) to build the network for expansion
        :param output_size: (int) how many to have at the end in total
        """
        super().__init__()
        self.output_size = output_size
        self.network = build_fc_network(configs).Model

    def forward(self, x: Tensor1D) -> Tensor1D:
        """selects the best values of x depending on the top values of the
        network output.

        :param x: (Tensor) the vectors to be expanded
        :return: (Tensor) after pruning and removing far from desired
        """
        B, _, S = x.shape
        x = self.network(x.view(B, -1))
        return x.view(B, self.output_size, S)


# Functions to build various modules and return BuildOutput type #############


def get_activation_function(
    activation: ActivationType, n_dims: int = None, epsilon: float = default_epsilon
) -> t.Optional[nn.Module]:
    match n_dims:
        case 2:
            softmax_module = nn.Softmax2d
        case _:
            softmax_module = nn.Softmax

    match activation:
        case ActivationType.ReLU:
            return ComplexAct(nn.ReLU, inplace=True)
        case ActivationType.Softmax:
            return ComplexAct(softmax_module, dim=1)
        case ActivationType.Sigmoid:
            return ComplexAct(nn.Sigmoid)
        case ActivationType.Tanh:
            return ComplexAct(nn.Tanh)
        case ActivationType.ModReLU:
            return CplxModReLU(threshold=0.5)
        case ActivationType.AdaModReLU:
            return CplxAdaptiveModReLU()
        case ActivationType.ZReLU:
            return ZReLU()
        case ActivationType.CReLU:
            return ComplexReLU()
        case ActivationType.Retract:
            return Retract()
        case ActivationType.Norm:
            return Normalize(epsilon=epsilon)
        case ActivationType.NoActivation | None:
            return None
        case _:
            raise NotImplementedError(f"{activation} is not implemented")


def get_normalization_layer(
    normalization: NormalizationType, n_dims: int, num_features: int, affine: bool
) -> t.Optional[nn.Module]:
    match normalization:
        case NormalizationType.NaiveNorm:
            layer = NaiveComplexBatchNorm1d if n_dims == 1 else NaiveComplexBatchNorm2d
        case NormalizationType.ComplexNorm:
            layer = ComplexBatchNorm1d if n_dims == 1 else ComplexBatchNorm2d
        case ActivationType.NoActivation | None:
            return None
        case _:
            raise NotImplementedError(f"{normalization} is not implemented")
    return layer(num_features=num_features, affine=affine)


def build_fc_norm_actv_layer(configs: FullyConnectedConfig, test: bool = True) -> BuildOutput:
    """Builds a single fc layer that contains normalization and activation
    depending on the configs.

    :param configs:
    :param test:
    :return:
    """
    fc_bias = (not configs.norm) and configs.bias
    modules = OrderedDict()
    if configs.scale_input:
        modules.update({"scale_input": Scale(configs.scale_input)})
    modules.update(
        {
            "FC": nn.Linear(
                in_features=configs.in_channels, out_features=configs.out_channels, bias=fc_bias, dtype=torch.complex64
            ),
        }
    )
    if configs.norm:
        norm = get_normalization_layer(configs.norm, n_dims=1, num_features=configs.out_channels, affine=configs.bias)
        if norm:
            modules.update({"norm": norm})

    if configs.dropout and configs.dropout > 0:
        assert 0 <= configs.dropout <= 1, configs.dropout
        modules.update({"dropout": nn.Dropout(p=configs.dropout)})

    if configs.activation:
        activation = get_activation_function(configs.activation)
        if activation:
            modules.update({"activation": activation})
    if configs.scale_output:
        modules.update({"scale_output": Scale(configs.scale_output)})
    model = nn.Sequential(modules)
    if test:
        x = torch.rand([2, configs.in_channels], dtype=torch.complex64)
        x = model(x)
        assert x.shape[1] == configs.out_channels
        return BuildOutput(model, x.shape)
    return BuildOutput(model, None)


def build_fc_network(
    configs: t.Sequence[FullyConnectedConfig], pre_squeeze: bool = False, test: bool = True
) -> BuildOutput:
    """Builds a consecutive layers of fully connected layers (that each may
    contain normalization and activation)

    :param configs:
    :param pre_squeeze:
    :param test:
    :return: (BuildOutput)
    """
    modules = OrderedDict()
    if pre_squeeze:
        modules.update({"Squeeze": Squeeze(dims=[2, 3])})
    for i, c in enumerate(configs):
        modules.update({f"FC_{i}": build_fc_norm_actv_layer(c).Model})
    model = nn.Sequential(modules)

    if test:
        x = torch.rand([2, configs[0].in_channels], dtype=torch.complex64)
        if pre_squeeze:
            x = x.unsqueeze(dim=2).unsqueeze(dim=3)
        x = model(x)
        return BuildOutput(model, x.shape)
    return BuildOutput(model, None)


def build_conv(configs: ConvConfig, test: bool = True) -> BuildOutput:
    """Builds a single convolution layer that could be 1D or 2D."""
    match configs.n_dims:
        case 1:
            dim = "1d"
            conv_module = nn.Conv1d
            avgpool_module = nn.AvgPool1d
            maxpool_module = nn.MaxPool1d
        case 2:
            dim = "2d"
            conv_module = nn.Conv2d
            avgpool_module = nn.AvgPool2d
            maxpool_module = nn.MaxPool2d
        case _:
            raise NotImplementedError(f"We currently support 1D or 2D, given {configs.n_dims}")

    conv_bias = (not configs.norm) and configs.bias
    assert (
        not configs.downsample or not configs.upsample
    ), "At most, we should have one resampling layer (either upsample or downsample"
    modules = OrderedDict()
    if configs.scale_input:
        modules.update({"scale_input": Scale(configs.scale_input)})
    modules.update(
        {
            f"conv{dim}": conv_module(
                in_channels=configs.in_channels,
                out_channels=configs.out_channels,
                kernel_size=configs.kernel_size,
                stride=configs.stride,
                padding=configs.padding,
                dilation=configs.dilation,
                bias=conv_bias,
                dtype=torch.complex64,
            ),
        }
    )
    if configs.norm:
        norm = get_normalization_layer(
            configs.norm, n_dims=configs.n_dims, num_features=configs.out_channels, affine=configs.bias
        )
        if norm:
            modules.update({f"norm{dim}": norm})

    if configs.activation:
        activation = get_activation_function(configs.activation, configs.n_dims)
        if activation:
            modules.update({"activation": activation})

    if configs.upsample:
        modules.update(
            {"upsample": nn.Upsample(scale_factor=configs.upsample.scale_factor, mode=configs.upsample.type)}
        )

    if configs.downsample:
        match configs.downsample.type:
            case PoolType.MaxPool:
                modules.update(
                    {
                        "downsample": maxpool_module(
                            kernel_size=configs.downsample.kernel,
                            stride=configs.downsample.stride,
                            padding=configs.downsample.padding,
                        )
                    }
                )
            case PoolType.AvgPool:
                modules.update(
                    {
                        "downsample": avgpool_module(
                            kernel_size=configs.downsample.kernel,
                            stride=configs.downsample.stride,
                            padding=configs.downsample.padding,
                        )
                    }
                )
            case _:
                raise NotImplementedError(f"Unsupported type for downsampling, given: {configs.downsample.type}")
    if configs.scale_output:
        modules.update({"scale_output": Scale(configs.scale_output)})
    model = nn.Sequential(modules)
    if test:
        dims = [1024] * configs.n_dims
        x = torch.rand([2, configs.in_channels, *dims], dtype=torch.complex64)
        x = model(x)
        assert x.shape[1] == configs.out_channels
        return BuildOutput(model, x.shape)
    return BuildOutput(model, None)


def build_conv_network(configs: t.Sequence[ConvConfig], test: bool = True) -> BuildOutput:
    """Builds consecutive layers of convolution layers.

    :param configs:
    :param test:
    :return:
    """
    modules = OrderedDict()
    for i, c in enumerate(configs):
        modules.update({f"Conv_{i}": build_conv(c).Model})
    model = nn.Sequential(modules)

    if test:
        x = torch.rand([2, configs[0].in_channels, 224, 224], dtype=torch.complex64)
        x = model(x)
        return BuildOutput(model, x.shape)
    return BuildOutput(model, None)


def build_dense_block(configs: t.Sequence[ConvConfig], test: bool = True) -> BuildOutput:
    """Builds a dense block and performs testing.

    :param configs:
    :param test:
    :return:
    """
    model = DenseBlock(configs)

    if test:
        x = torch.rand([2, configs[0].in_channels, 224, 224], dtype=torch.complex64)
        x = model(x)
        return BuildOutput(model, x.shape)
    return BuildOutput(model, None)


def build_dense_network(configs: t.Sequence[t.Sequence[ConvConfig]], test: bool = True) -> BuildOutput:
    """builds a dense network which is a consecutive dense and transition
    blocks.

    :param configs:
    :param test:
    :return:
    """
    modules = OrderedDict()
    for i, c in enumerate(configs):
        if i % 2:
            modules.update({f"Dense_{i}": build_dense_block(c).Model})
        else:
            modules.update({f"Transition_{i}": build_conv_network(c).Model})
    model = nn.Sequential(modules)

    if test:
        x = torch.rand([2, configs[0][0].in_channels, 224, 224], dtype=torch.complex64)
        x = model(x)
        return BuildOutput(model, x.shape)
    return BuildOutput(model, None)

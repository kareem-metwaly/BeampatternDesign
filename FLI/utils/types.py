import typing as t
from pathlib import Path

import pydantic
from torch import Tensor, nn

DirectoryPath = t.Union[str, Path] if t.TYPE_CHECKING else pydantic.DirectoryPath
FilePath = t.Union[str, Path] if t.TYPE_CHECKING else pydantic.FilePath

# defines a scaler tensors, like `loss`
TensorScaler = t.TypeVar("TensorScaler", bound=Tensor)

# defines 1D tensors, like `waveform` when it's vectorized to be of size MNx1;
# note it is not necessarily of shape 1D as it might be batched
Tensor1D = t.TypeVar("Tensor1D", bound=Tensor)

# defines 2D tensors, like `beampattern` of size NxK; note that it is not actually of shape 2D as it might be batched
Tensor2D = t.TypeVar("Tensor2D", bound=Tensor)

# defines 3D tensors, like `A`, `F` and `H`; note that it is not actually of shape 3D as it might be batched
Tensor3D = t.TypeVar("Tensor3D", bound=Tensor)

ActivationModule = t.Union[nn.Softmax, nn.Softmax2d, nn.ReLU, nn.Sigmoid, nn.Tanh]

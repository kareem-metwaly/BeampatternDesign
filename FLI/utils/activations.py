"""Originally from https://github.com/ivannz/cplxmodule Check the repo for more
details."""

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.modules import Module
from utils.complex_functions import complex_relu


def modrelu(inp: Tensor, threshold: Parameter = 0.5) -> Tensor:
    r"""Soft-threshold the modulus of the complex tensor.

    Parameters
    ----------
    inp : tensor
        The complex valued data to which modReLU is to be applied elementwise.

    threshold : float, default=0.5
        The clipping threshold of this version of modReLU. See details.

    Returns
    -------
    output : tensor
        The values which have their complex modulus soft-thresholded and their
        complex phase retained.

    Details
    -------
    This function actually implements a slightly reparameterized version
    of modReLU (note the negative sign):

    $$
    \operatorname{modReLU}
        \colon \mathbb{C} \times \mathbb{R} \to  \mathbb{C}
        \colon
            (z, \tau)
                \mapsto z \max \biggl\{
                    0, 1 - \frac{\tau}{\lvert z \rvert}
                \biggr\}
                = \max \biggl\{
                    0, \lvert z \rvert - \tau
                \biggr\} e^{j \phi}
        \,. $$

    This parameterization deviates from the non-linearity, originally proposed
    in

        [Arjovsky et al. (2016)](http://proceedings.mlr.press/v48/arjovsky16.html)

    by having the sign of the bias parameter $b$ in eq.~(8) FLIPPED.

    The rationale behind this discrepancy in the implementation was that this
    non-linearity resembles the soft-thresholding operator, with the parameter
    playing the role of the zeroing threshold:

        the higher the threshold the larger should the magnitude of the complex
        number be in order to avoid being zeroed.
    """
    modulus = torch.clamp(abs(inp), min=1e-5)
    return inp * torch.relu(1.0 - threshold / modulus)


class CplxModReLU(Module):
    r"""Applies soft thresholding to the complex modulus:
    $$
        F
        \colon \mathbb{C} \to \mathbb{C}
        \colon z \mapsto (\lvert z \rvert - \tau)_+
                         \tfrac{z}{\lvert z \rvert}
        \,, $$
    with $\tau \in \mathbb{R}$. The if threshold=None then it
    becomes a learnable parameter.
    """

    threshold: Parameter

    def __init__(self, threshold=0.5):
        super().__init__()
        if not isinstance(threshold, float):
            threshold = Parameter(torch.rand(1) * 0.25)
        self.threshold = threshold

    def forward(self, inp: Tensor) -> Tensor:
        return modrelu(inp, self.threshold)


class CplxAdaptiveModReLU(Module):
    r"""Applies soft thresholding to the complex modulus:
    $$
        F
        \colon \mathbb{C}^d \to \mathbb{C}^d
        \colon z \mapsto (\lvert z_j \rvert - \tau_j)_+
                        \tfrac{z_j}{\lvert z_j \rvert}
        \,, $$
    with $\tau_j \in \mathbb{R}$ being the $j$-th learnable threshold. Torch's
    broadcasting rules apply and the passed dimensions must conform with the
    upstream input. `CplxChanneledModReLU(1)` learns a common threshold for all
    features of the $d$-dim complex vector, and `CplxChanneledModReLU(d)` lets
    each dimension have its own threshold.
    """

    threshold: Parameter

    def __init__(self, *dim):
        super().__init__()
        self.dim = dim if dim else (1,)
        self.threshold = torch.nn.Parameter(torch.randn(*self.dim) * 0.02)

    def forward(self, inp: Tensor):
        return modrelu(inp, self.threshold)

    def __repr__(self):
        body = repr(self.dim)[1:-1] if len(self.dim) > 1 else repr(self.dim[0])
        return f"{self.__class__.__name__}({body})"


class Magnitude(Module):
    """retrieves the magnitude as real value from the input complex value."""

    @staticmethod
    def forward(inp: Tensor) -> Tensor:
        """retrieves the magnitude."""
        return inp.abs()


def zrelu(inp: Tensor) -> Tensor:
    """computes zReLU."""
    angle = inp.angle
    return torch.where(0 <= angle <= (torch.pi / 2), inp, 0)


class ZReLU(Module):
    """computes the zReLU activation which is identity only if theta between 0
    and pi/2, o.w.

    o/p is 0
    """

    @staticmethod
    def forward(inp: Tensor) -> Tensor:
        """compute zReLU."""
        return zrelu(inp)


class ComplexReLU(Module):
    """ReLU for complex."""

    @staticmethod
    def forward(inp):
        return complex_relu(inp)

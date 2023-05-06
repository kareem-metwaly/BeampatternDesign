from torch.nn import Module
from utils.complex_functions import complex_dropout, complex_dropout2d


class ComplexDropout(Module):
    """perform 1D complex dropout."""

    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            return complex_dropout(x, self.p)
        else:
            return x


class ComplexDropout2d(Module):
    """2D complex dropout module."""

    def __init__(self, p=0.5):
        super(ComplexDropout2d, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            return complex_dropout2d(x, self.p)
        else:
            return x

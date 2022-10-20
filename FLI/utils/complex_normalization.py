#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Tue Mar 19 10:30:02 2019.

originally created by: Sebastien M. Popoff

modified to fix memory leak issue with complex batch normalization

Based on https://openreview.net/forum?id=H1T2hmZAb
"""

import torch
from torch.nn import BatchNorm1d, BatchNorm2d, Module, Parameter, init


class NaiveComplexBatchNorm1d(Module):
    """Naive approach to complex batch norm, perform batch norm independently
    on real and imaginary part."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(NaiveComplexBatchNorm1d, self).__init__()
        self.bn_r = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        return self.bn_r(x.real).type(torch.complex64) + 1j * self.bn_i(x.imag).type(torch.complex64)


class NaiveComplexBatchNorm2d(Module):
    """Naive approach to complex batch norm, perform batch norm independently
    on real and imaginary part."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(NaiveComplexBatchNorm2d, self).__init__()
        self.bn_r = BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        return self.bn_r(x.real).type(torch.complex64) + 1j * self.bn_i(x.imag).type(torch.complex64)


class _ComplexBatchNorm(Module):
    running_mean: torch.Tensor

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features, dtype=torch.complex64))
            self.register_buffer("running_covar", torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_covar", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):
    def forward(self, x):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean = x.real.mean([0, 2, 3]).type(torch.complex64) + 1j * x.imag.mean([0, 2, 3]).type(torch.complex64)
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                )

        x = x - mean[None, :, None, None]
        n = x.numel() / x.size(1)

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            Crr = 1.0 / n * x.real.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1.0 / n * x.imag.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (x.real.mul(x.imag)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                    exponential_average_factor * Crr * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                    exponential_average_factor * Cii * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 1]
                )

                self.running_covar[:, 2] = (
                    exponential_average_factor * Cri * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        x = (Rrr[None, :, None, None] * x.real + Rri[None, :, None, None] * x.imag).type(torch.complex64) + 1j * (
            Rii[None, :, None, None] * x.imag + Rri[None, :, None, None] * x.real
        ).type(torch.complex64)
        if self.affine:
            x = (
                self.weight[None, :, 0, None, None] * x.real
                + self.weight[None, :, 2, None, None] * x.imag
                + self.bias[None, :, 0, None, None]
            ).type(torch.complex64) + 1j * (
                self.weight[None, :, 2, None, None] * x.real
                + self.weight[None, :, 1, None, None] * x.imag
                + self.bias[None, :, 1, None, None]
            ).type(
                torch.complex64
            )

        return x


class ComplexBatchNorm1d(_ComplexBatchNorm):
    def forward(self, x):

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            mean = x.real.mean(dim=0).type(torch.complex64) + 1j * x.imag.mean(dim=0).type(torch.complex64)
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                )

        x = x - mean[None, ...]
        n = x.numel() / x.size(1)
        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            Crr = x.real.var(dim=0, unbiased=False) + self.eps
            Cii = x.imag.var(dim=0, unbiased=False) + self.eps
            Cri = (x.real.mul(x.imag)).mean(dim=0)
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                    exponential_average_factor * Crr * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                    exponential_average_factor * Cii * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 1]
                )

                self.running_covar[:, 2] = (
                    exponential_average_factor * Cri * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        x = (Rrr[None, :] * x.real + Rri[None, :] * x.imag).type(torch.complex64) + 1j * (
            Rii[None, :] * x.imag + Rri[None, :] * x.real
        ).type(torch.complex64)

        if self.affine:
            x = (self.weight[None, :, 0] * x.real + self.weight[None, :, 2] * x.imag + self.bias[None, :, 0]).type(
                torch.complex64
            ) + 1j * (self.weight[None, :, 2] * x.real + self.weight[None, :, 1] * x.imag + self.bias[None, :, 1]).type(
                torch.complex64
            )

        return x

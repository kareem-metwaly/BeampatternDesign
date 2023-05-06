"""implements any methods related to the physics of the problem.

for example, it implements find_matrix_A.
"""
# import math

import torch
from utils.config_classes import Parameters
from utils.types import Tensor1D, Tensor2D, Tensor3D


def find_matrix_A(params: Parameters) -> Tensor3D:
    """calculates matrix A, related to the angles.

    :param params: (Parameters) of the physical problem
    :return: (Tensor) of size K x M x N
    """
    A = torch.zeros([params.K, params.M, params.N], dtype=torch.complex64)
    dist = 2 * (params.B / 2 + params.fc)
    for N_idx, freq in enumerate(torch.arange(-params.N / 2, params.N / 2)):
        for K_idx, theta in enumerate(torch.arange(0, params.K) * torch.pi / 180):
            for M_idx in range(params.M):
                A[K_idx, M_idx, N_idx] = torch.exp(
                    -1j * 2 * torch.pi * (params.fc + freq / (params.N * params.Ts)) * M_idx * torch.cos(theta) / dist
                )
    return A


def find_matrix_F(params: Parameters) -> Tensor3D:
    """calculates matrix F, related to Fourier transform.

    :param params: (Parameters) of the physical problem
    :return: (Tensor) of size M x M.N x N
    """
    F = torch.zeros([params.N, params.N], dtype=torch.complex64)
    for N_idx, n in enumerate(torch.arange(0, 1, 1 / params.N)):
        for L_idx, p in enumerate(torch.arange(-params.N / 2, params.N / 2)):
            F[N_idx, L_idx] = torch.exp(1j * 2 * torch.pi * n * p)

    Fb = torch.zeros([params.M, params.M * params.N, params.N], dtype=torch.complex64)
    Identity = torch.eye(params.M)
    for N_idx in range(params.N):
        Fb[..., N_idx] = torch.kron(Identity, F[:, N_idx].conj())
    return Fb


def find_matrix_H_using_AF(A: Tensor3D, F: Tensor3D) -> Tensor3D:
    """calculates matrix H from A and F.

    :param A: (Tensor) of size K x M x N
    :param F: (Tensor) of size M x M.N x N
    :return: (Tensor) of size K x N x M.N
    """
    K, M, N = A.size()
    H = torch.zeros([K, N, M * N], dtype=torch.complex64)
    for n in range(N):
        H[:, n, :] = A[..., n].matmul(F[..., n])
    return H


def find_matrix_H_using_params(params: Parameters) -> Tensor3D:
    """calculates matrix H from parameters directly.

    :param params: (Parameters) of the physical problem
    :return: (Tensor) of size K x N x M.N
    """
    A = find_matrix_A(params)
    F = find_matrix_F(params)
    return find_matrix_H_using_AF(A=A, F=F)


def find_matrix_P_using_params(params: Parameters) -> Tensor2D:
    """Calculates the matrix P which is H^H .

    H i.e. F^H . A^H . A . F given parameters
    """
    H = find_matrix_H_using_params(params)  # of dimensions K, N, M.N
    H = H.flatten(start_dim=0, end_dim=1)
    P = H.H.matmul(H)  # finding hermitian and multiplying with H
    return P


def find_matrix_P_using_H(H: Tensor3D) -> Tensor2D:
    """Calculates the matrix P which is H^H .

    H i.e. F^H . A^H . A . F given input H which is of size K x N x M.N
    """
    H = H.flatten(start_dim=0, end_dim=1)
    P = H.H.matmul(H)  # finding hermitian and multiplying with H
    return P


def calculate_diff_dB(predictions: Tensor2D, groundtruths: Tensor2D) -> Tensor1D:
    """Similar to the Matlab implementation, it generates the dB error between
    predictions and groundtruths.

    :param Tensor2D predictions: the predicted beampattern as a batch B x K x N
    :param Tensor2D groundtruths: the desired beampattern as a batch B x K x N
    :return: TensorScaler a batch of values (size B) representing error for each prediction/groundtruth pair in dB
    """
    return predictions.abs().sub(groundtruths.abs()).pow(2).sum(dim=(1, 2)).log10().mul(10)
    # .div(groundtruths.real.max(dim=2).values.max(dim=1).values.clamp(min=1e-4))

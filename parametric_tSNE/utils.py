from __future__ import division  # Python 2 users only
from __future__ import print_function

__doc__ = """ Utility functions used for tSNE modelling"""

import numpy as np
import torch


def Hbeta_vec(distances: np.ndarray, betas: np.ndarray):
    """
    Function that computes the Gaussian kernel values given a vector of
    squared Euclidean distances, and the precision of the Gaussian kernel.
    The function also computes the perplexity of the distribution.
    From Parametric t-SNE for matlab at https://lvdmaaten.github.io/tsne/
    Parameters
    ----------
    distances: 2-d array_like, (N,N)
        Square matrix of distances between data points
    betas: 1-d array_like, (N,)
        Vector of precisions of the Gaussian kernel. beta = (2 sigma**2)^-1
    Returns
    -------
    H: 1-d array_like, (N,)
        Entropy of each point
    p_matr: 2-d array_like, (N,N)
        array of probability values
        The scalar formula for p_matr is:
        p_matr = np.exp(-D * beta) / sum(np.exp(-D * beta))
        This funcion is vectorized and calculates the full P matrix
    """
    beta_matr = betas[:, np.newaxis] * np.ones_like(distances)
    p_matr = np.exp(-distances * beta_matr)
    sumP = np.sum(p_matr, axis=1)
    H = np.log(sumP) + (betas * np.sum(distances * p_matr, axis=1)) / sumP
    p_matr = p_matr / (sumP[:, np.newaxis] * np.ones_like(p_matr))

    return H, p_matr


def Hbeta_scalar(distances: np.ndarray, beta: float):
    """
    Function that computes the Gaussian kernel values given a vector of
    squared Euclidean distances, and the precision of the Gaussian kernel.
    The function also computes the perplexity of the distribution.
    From Parametric t-SNE for matlab at https://lvdmaaten.github.io/tsne/
    Parameters
    ----------
    distances: 1-d array_like, (N,)
        Distance between the current data point and all others
    beta: float
        Precision of the Gaussian kernel. beta = (2 sigma**2)^-1
    Returns
    -------
    H: float
        Entropy
    p_matr: 1-d array_like, (N,)
        array of probability values
        p_matr = np.exp(-D * beta) / sum(np.exp(-D * beta))
    """

    p_matr = np.exp(-distances * beta)
    sumP = np.sum(p_matr)
    H = np.log(sumP) + (beta * np.sum(distances * p_matr)) / sumP
    p_matr = p_matr / sumP

    return H, p_matr


def get_squared_cross_diff_np(x: np.ndarray):
    """Compute squared differences of sample data vectors.
        Z_ij = ||x_i - x_j||^2, where x_i = X_[i, :]
    Parameters
    ----------
    x : 2-d array_like, (N, D)
        Calculates outer vector product
        This is the current batch of input data; `batch_size` x `dimension`
    Returns
    -------
    Z_ij: 2-d array_like, (N, N)
        `batch_size` x `batch_size`
        Matrix of squared differences between x_i and x_j
    """
    batch_size = x.shape[0]

    expanded = np.expand_dims(x, 1)
    # "tiled" is now stacked up all the samples along dimension 1
    tiled = np.tile(expanded, np.stack([1, batch_size, 1]))

    tiled_trans = np.transpose(tiled, axes=[1, 0, 2])

    diffs = tiled - tiled_trans
    sum_act = np.sum(np.square(diffs), axis=2)

    return sum_act


def get_squared_cross_diff_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Compute squared differences of sample data vectors in PyTorch.
    Z_ij = ||x_i - x_j||^2, where x_i = X_[i, :]

    Parameters
    ----------
    x : 2-d tensor, (N, D)
        This is the current batch of input data; `batch_size` x `dimension`

    Returns
    -------
    Z_ij: 2-d tensor, (N, N)
        `batch_size` x `batch_size`
        Matrix of squared differences between x_i and x_j
    """
    # Using broadcasting for efficient computation of pairwise differences
    diffs = x.unsqueeze(1) - x.unsqueeze(0)
    sum_act = torch.sum(diffs ** 2, dim=2)
    
    return sum_act


def get_Lmax(num_points) -> int:
    return np.floor(np.log2(num_points / 4.0))


def get_multiscale_perplexities(num_points: int) -> np.ndarray:
    """
    Generate range of perplexities based on number of points.
    
    Parameters
    ----------
    num_points: int
        Number of known data points

    Returns
    -------
    perplexities: np.ndarray
        List of perplexities to use

    References
    ----------
    Lee, John & Peluffo-Ordóñez, Diego & Verleysen, Michel. (2015).
    Multiscale stochastic neighbor embedding: Towards parameter-free dimensionality reduction.
    """
    Lmax = get_Lmax(num_points)
    l_vals = np.arange(2, Lmax)
    perplexities = 2.0 ** l_vals
    return perplexities


def calc_betas_loop(
    indata: np.ndarray, perplexity: float, tol: float = 1e-4, max_tries: int = 50
):
    """
    Calculate beta values for a desired perplexity via binary search
    Uses a loop; could be made faster with cython
    In my tests, vectorized version (calc_betas_vec) doesn't end up much faster
    likely due to higher memory usage
    Parameters
    ----------
    indata: 2-d array_like, (N,D)
    perplexity: float
        Desired perplexity. See literature on tSNE for details.
        Roughly speaking, this is the number of points which should be considered
        when calculating distances
    tol: float, optional
        Absolute tolerance in the entropy for calculating the beta values
        Once entropy stops shifting by this amount the search stops
    max_tries: int, optional
        Maximum number of iterations to use. Default 50.
    Returns
    -------
    betas: 1-D array_like, (N,)
        Calculated beta values
    Hs: 1-d array_like, (N,)
        Entropy at each point given the
    p_matr: 2-d array_like, (N,N)
        Probability matrix between each pair of points
    """
    logPx = np.log(perplexity)
    num_samps = indata.shape[0]

    beta_init = np.ones([num_samps], dtype=float)
    betas = beta_init.copy()
    p_matr = np.zeros([num_samps, num_samps])
    Hs = beta_init.copy()

    in_sq_diffs = get_squared_cross_diff_np(indata)

    loop_samps = range(num_samps)
    for ss in loop_samps:
        betamin = -np.inf
        betamax = np.inf

        Di = in_sq_diffs[ss, :]
        H, thisPx = Hbeta_scalar(Di, betas[ss])
        del H
        Hdiff = 100 * tol

        tries = 0
        while abs(Hdiff) > tol and tries < max_tries:
            # Compute the Gaussian kernel and entropy for the current precision
            H, thisPx = Hbeta_scalar(Di, betas[ss])
            Hdiff = H - logPx
            tries = tries + 1

            if Hdiff > 0.0:
                betamin = betas[ss]
                if np.isinf(betamax):
                    betas[ss] = betas[ss] * 2.0
                else:
                    betas[ss] = (betas[ss] + betamax) / 2.0
            else:
                betamax = betas[ss]
                if np.isinf(betamin):
                    betas[ss] = betas[ss] / 2.0
                else:
                    betas[ss] = (betas[ss] + betamin) / 2.0

        # Set the final row of P
        p_matr[ss, :] = thisPx
        Hs[ss] = H

    return betas, Hs, p_matr


def torch_set_diag(x: torch.Tensor, val: float) -> torch.Tensor:
    """
    Set diagonal of a matrix to by a new value
    Parameters
    ----------
    x: 2-d tensor, (N, M)
        Matrix to modify
    val: float
        Value to set diagonal to
    Returns
    -------
    x: 2-d tensor, (N, M)
        Modified matrix
    """
    x[torch.arange(x.shape[0]), torch.arange(x.shape[1])] = val
    return x
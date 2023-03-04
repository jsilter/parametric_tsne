#!/usr/bin/python
from __future__ import division  # Python 2 users only
from __future__ import print_function

__doc__ = """ 
Module for building a parametric tSNE model. 
Trains a neural network on input data.
One can then transform other data based on this model

Common abbreviations and terms:
N = number of points
D = number of dimensions
P  = number of perplexity values used.
If using multiple perplexities, some ndarrays will have a dimension of values
for each perplexity.

Main reference:
van der Maaten, L. (2009). Learning a parametric embedding by preserving local structure. RBM, 500(500), 26.
See README.md for others
"""

import datetime
import functools
from typing import List, Union, Optional, Iterator, Tuple

import numpy as np
import tensorflow as tf

from .utils import calc_betas_loop, get_squared_cross_diff_np

tfk = tf.keras
tfkl = tfk.layers

DEFAULT_EPS = 1e-7


def _make_P_ji(data: np.ndarray, betas: np.ndarray, in_sq_diffs: np.ndarray = None):
    """Calculate similarity probabilities based on data data
    Parameters
    ----------
    data : 2d np.ndarray, (N, D)
        Input data which we wish to calculate similarity probabilities
    betas : 2d np.ndarray, (N, P)
        Gaussian kernel used for each point.
    in_sq_diffs: 2d np.ndarray, (N, N)
        Squared difference between each data point. Can be reused to save computation,
        or if one wants a custom distance metric. Otherwise we use `get_squared_cross_diff_np`.
    Returns
    -------
    P_ji : 2d array_like, (N, N, P)
        Similarity probability matrix
    """
    if not in_sq_diffs:
        in_sq_diffs = get_squared_cross_diff_np(data)
    tmp = in_sq_diffs[:, :, np.newaxis] * betas[np.newaxis, :, :]
    P_ji = np.exp(-1.0 * tmp)
    return P_ji


def _make_P_np(data: np.ndarray, betas: np.ndarray):
    """
    Calculate similarity probabilities based on data data
    Parameters
    ----------
    data : 2d array_like, (N, D)
        Input data which we wish to calculate similarity probabilities
    betas : 2d array_like, (N, P)
        Gaussian kernel(s) used for each point.
    Returns
    -------
    P : nd array_like, (N, N, P)
        Symmetric similarity probability matrix
        Beta-values across third dimension
    """
    P_ji = _make_P_ji(data, betas)
    P_3 = np.zeros_like(P_ji)
    for zz in range(P_3.shape[2]):
        P_3[:, :, zz] = _get_normed_sym_np(P_ji[:, :, zz])
    P_ = P_3
    return P_


def _make_P_tf(data: tf.Tensor, betas: np.ndarray, batch_size: int):
    """Tensorflow implementation of _make_P_np.
    Not documented because not used, for example only."""
    in_sq_diffs = _get_squared_cross_diff_tf(data)
    tmp = in_sq_diffs * betas
    P = tf.exp(-1.0 * tmp)
    P = _get_normed_sym_tf(P, batch_size)
    return P


def _get_squared_cross_diff_tf(x: tf.Tensor):
    """Compute squared differences of sample data vectors.
    Implementation for Tensorflow Tensors
    Z_ij = ||x_i - x_j||^2, where x_i = X_[i, :]
    Parameters
    ----------
    x : 2-d Tensor, (N, D)
        Calculates outer vector product
        This is the current batch of input data; `batch_size` x `dimension`
    Returns
    -------
    Z_ij: 2-d Tensor, (N, N)
        `batch_size` x `batch_size`
        Tensor of squared differences between x_i and x_j
    """
    batch_size = tf.shape(x)[0]

    expanded = tf.expand_dims(x, 1)
    # "tiled" is now stacked up all the samples along dimension 1
    tiled = tf.tile(expanded, tf.stack([1, batch_size, 1]))

    tiled_trans = tf.transpose(tiled, perm=[1, 0, 2])

    diffs = tiled - tiled_trans
    sum_act = tf.reduce_sum(tf.square(diffs), axis=2)

    return sum_act


def _get_normed_sym_np(x: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    """
    Compute the normalized and symmetrized probability matrix from
    relative probabilities x, where x is a numpy array
    Parameters
    ----------
    x : 2-d array_like (N, N)
        asymmetric probabilities. For instance, x(i, j) = P(i|j)
    eps: float, optional
        Factor in denominator to prevent dividing by 0.
    Returns
    -------
    P : 2-d array_like (N, N)
        symmetric probabilities, making the assumption that P(i|j) = P(j|i)
        Diagonals are all 0s."""
    batch_size = x.shape[0]
    zero_diags = 1.0 - np.identity(batch_size)
    P = x * zero_diags
    norm_facs = np.sum(x, axis=0, keepdims=True)
    P = P / (norm_facs + eps)
    P = 0.5 * (P + np.transpose(P))

    return P


def _get_normed_sym_tf(x: tf.Tensor, batch_size: int) -> np.ndarray:
    """
    Compute the normalized and symmetrized probability matrix from
    relative probabilities x, where x is a Tensorflow Tensor
    Parameters
    ----------
    x : 2-d Tensor (N, N)
        asymmetric probabilities. For instance, x(i, j) = P(i|j)
    batch_size: int
        N dimension of `x.` Needs to be passed explicitly.
    Returns
    -------
    P : 2-d Tensor (N, N)
        symmetric probabilities, making the assumption that P(i|j) = P(j|i)
        Diagonals are all 0s."""
    toset = tf.constant(0, shape=[batch_size], dtype=x.dtype)
    x = tf.linalg.set_diag(x, toset)
    norm_facs = tf.reduce_sum(x, axis=0, keepdims=True)
    x = x / norm_facs
    x = 0.5 * (x + tf.transpose(x))

    return x


def _make_Q(output: tf.Tensor, alpha: float, batch_size: int):
    """
    Calculate the "Q" probability distribution of the output
    Based on the t-distribution.

    Parameters
    ----------
    output : 2-d Tensor (N, output_dims)
        Output of the neural network
    alpha : float
        `alpha` parameter. Recommend `output_dims` - 1.0
    batch_size : int
        N dimension of `output`.
    Returns
    -------
    Q_ : 2-d Tensor (N, N)
        Symmetric "Q" probability distribution; similarity of
        points based on output data
    """
    out_sq_diffs = _get_squared_cross_diff_tf(output)
    Q = tf.pow((1 + out_sq_diffs / alpha), -(alpha + 1) / 2)
    Q = _get_normed_sym_tf(Q, batch_size)
    return Q


def kl_loss(
    y_true: np.ndarray,
    y_pred: tf.Tensor,
    alpha: float = 1.0,
    batch_size: int = None,
    num_perplexities: int = None,
    eps: float = DEFAULT_EPS,
):
    """Kullback-Leibler Loss function (Tensorflow)
    between the "true" output and the "predicted" output
    Parameters
    ----------
    y_true : 2d array_like (N, N*P)
        Should be the P matrix calculated from input data.
        Differences in data points using a Gaussian probability distribution
        Different P (perplexity) values stacked along dimension 1
    y_pred : tf.Tensor, 2d (N, output_dims)
        Output of the neural network. We will calculate
        the Q matrix based on this output
    alpha : float, optional
        Parameter used to calculate Q. Default 1.0
    batch_size : int, required
        Number of samples per batch. y_true.shape[0]
    num_perplexities : int, required
        Number of perplexities stacked along axis 1
    eps: float, optional
        Epsilon used to prevent divide by zero
    Returns
    -------
    kl_loss : tf.Tensor, scalar value
        Kullback-Leibler divergence P_ || Q_

    """
    P_ = y_true
    Q = _make_Q(y_pred, alpha, batch_size)

    tf_eps = tf.constant(eps, dtype=P_.dtype)

    kls_per_beta = []
    components = tf.split(P_, num_perplexities, axis=1, name="split_perp")
    for cur_beta_P in components:
        # yrange = tf.range(zz*batch_size, (zz+1)*batch_size)
        # cur_beta_P = tf.slice(P_, [zz*batch_size, [-1, batch_size])
        # cur_beta_P = P_
        kl_matr = tf.multiply(
            cur_beta_P,
            tf.math.log(cur_beta_P + tf_eps) - tf.math.log(Q + tf_eps),
            name="kl_matr",
        )
        toset = tf.constant(0, shape=[batch_size], dtype=kl_matr.dtype)
        kl_matr_keep = tf.linalg.set_diag(kl_matr, toset)
        kl_total_cost_cur_beta = tf.reduce_sum(kl_matr_keep)
        kls_per_beta.append(kl_total_cost_cur_beta)
    kl_total_cost = tf.add_n(kls_per_beta)

    return kl_total_cost


class Parametric_tSNE:
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        perplexities: Union[float, List[float]],
        alpha: float = 1.0,
        optimizer="adam",
        all_layers: Optional[List[tfkl.Layer]] = None,
        do_pretrain: bool = True,
        seed: int = 0,
        batch_size: int = 64,
    ):
        """

        Main class used for training parametric tSNE model.

        num_inputs : int
            Dimension of the (high-dimensional) data
        num_outputs : int
            Dimension of the (low-dimensional) output
        perplexities:
            Desired perplexities (one or more). Generally interpreted as the number of neighbors to use
            for distance comparisons but actually doesn't need to be an integer.
            Can be an array for multi-scale.
        Roughly speaking, this is the number of points which should be considered
        when calculating distances between points. Can be None if one provides own training betas.
        alpha: float
            alpha scaling parameter of output t-distribution
        optimizer: string or Optimizer, optional
            default 'adam'. Passed to model.fit
        all_layers: Optional
            optional. Layers to use in model. If none provided, uses
            the same structure as van der Maaten 2009
        do_pretrain: bool, optional
            Whether to perform layerwise pretraining. Default True
        seed: int, optional
            Default 0. Seed for Tensorflow state.
        batch_size: int, optional.
            Default 64. Batch size used in training.
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        if perplexities is not None and not isinstance(
            perplexities, (list, tuple, np.ndarray)
        ):
            perplexities = np.array([perplexities])
        self.perplexities = perplexities
        self.num_perplexities = None
        if perplexities is not None:
            self.num_perplexities = len(np.array(perplexities))
        self.alpha = alpha
        self._optimizer = optimizer
        self._batch_size = batch_size
        self.do_pretrain = do_pretrain

        self._loss_func = None
        self._epochs = None
        self._training_betas = None

        tf.random.set_seed(seed)
        np.random.seed(seed)

        # If no layers provided, use the same architecture as van der maaten 2009 paper
        if all_layers is None:
            all_layer_sizes = [num_inputs, 500, 500, 2000, num_outputs]
            all_layers = [
                tfkl.Dense(
                    all_layer_sizes[1],
                    input_shape=(num_inputs,),
                    activation="sigmoid",
                    kernel_initializer="glorot_uniform",
                )
            ]

            for lsize in all_layer_sizes[2:-1]:
                cur_layer = tfkl.Dense(
                    lsize, activation="sigmoid", kernel_initializer="glorot_uniform"
                )
                all_layers.append(cur_layer)

            all_layers.append(
                tfkl.Dense(
                    num_outputs,
                    activation="linear",
                    kernel_initializer="glorot_uniform",
                )
            )

        self._all_layers = all_layers
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize Keras model"""
        self.model = tfk.models.Sequential(self._all_layers)

    @staticmethod
    def _calc_training_betas(
        training_data: np.ndarray,
        perplexities: Union[float, np.ndarray],
        beta_batch_size: int = 1000,
    ) -> np.ndarray:
        """
        Calculate beta values (gaussian kernel widths) used for training the model
        For memory reasons, only uses beta_batch_size points at a time.
        Parameters
        ----------
        training_data : 2d array_like, (N, D)
        perplexities : float or ndarray-like, (P,)
        beta_batch_size : int, optional
            Only use `beta_batch_size` points to calculate beta values. This is
            for speed and memory reasons. Data must be well-shuffled for this to be effective,
            betas will be calculated based on regular batches of this size
        Returns
        -------
        betas : 2D array_like (N,P)
            Beta values for each point and perplexity value
        """
        assert (
            perplexities is not None
        ), "Must provide desired perplexities if training beta values"
        num_pts = len(training_data)
        if not isinstance(perplexities, (list, tuple, np.ndarray)):
            perplexities = np.array([perplexities])
        num_perplexities = len(perplexities)
        training_betas = np.zeros([num_pts, num_perplexities])

        # To calculate betas, only use `beta_batch_size` points at a time
        cur_start = 0
        cur_end = min(cur_start + beta_batch_size, num_pts)
        while cur_start < num_pts:
            cur_training_data = training_data[cur_start:cur_end, :]

            for pind, curperp in enumerate(perplexities):
                cur_training_betas, cur_P, cur_Hs = calc_betas_loop(
                    cur_training_data, curperp
                )
                training_betas[cur_start:cur_end, pind] = cur_training_betas

            cur_start += beta_batch_size
            cur_end = min(cur_start + beta_batch_size, num_pts)

        return training_betas

    def _pretrain_layers(
        self,
        pretrain_data: np.ndarray,
        batch_size: int = 64,
        epochs: int = 10,
        verbose: int = 0,
    ):
        """
        Pretrain layers using stacked auto-encoders
        Parameters
        ----------
        pretrain_data : 2d array, (N,D)
            Data to use for pretraining. Can be the same as used for training
        batch_size : int, optional
        epochs : int, optional
        verbose : int, optional
            Verbosity level. Passed to Keras fit method
        Returns
        -------
            None. Layers trained in place
        """
        if verbose:
            print(
                "{time}: Pretraining {num_layers:d} layers".format(
                    time=datetime.datetime.now(), num_layers=len(self._all_layers)
                )
            )

        for ind, end_layer in enumerate(self._all_layers):
            # Create AE and training
            cur_layers = self._all_layers[0:(ind + 1)]
            ae = tfk.models.Sequential(cur_layers)

            decoder = tfkl.Dense(pretrain_data.shape[1], activation="linear")
            ae.add(decoder)

            ae.compile(loss="mean_squared_error", optimizer="rmsprop")
            ae.fit(
                pretrain_data,
                pretrain_data,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
            )

        self.model = tfk.models.Sequential(self._all_layers)

        if verbose:
            print("{time}: Finished pretraining".format(time=datetime.datetime.now()))

    def _init_loss_func(self):
        """Initialize loss function based on parameters fed to constructor
        Necessary to do this, so we can save/load the model using Keras, since
        the loss function is a custom object"""
        kl_loss_func = functools.partial(
            kl_loss,
            alpha=self.alpha,
            batch_size=self._batch_size,
            num_perplexities=self.num_perplexities,
        )
        kl_loss_func.__name__ = "KL-Divergence"
        self._loss_func = kl_loss_func

    @staticmethod
    def _get_num_perplexities(
        training_betas: Optional[np.ndarray], num_perplexities: Optional[int]
    ):
        if training_betas is None and num_perplexities is None:
            return None

        if training_betas is None:
            return num_perplexities
        elif training_betas is not None and num_perplexities is None:
            return training_betas.shape[1]
        else:
            if len(training_betas.shape) == 1:
                assert (
                    num_perplexities == 1
                ), "Mismatch between data training betas and num_perplexities"
            else:
                assert training_betas.shape[1] == num_perplexities
            return num_perplexities

    def fit(
        self,
        training_data: np.ndarray,
        training_betas: Optional[np.ndarray] = None,
        epochs: int = 10,
        verbose: int = 0,
    ):
        """
        Train the neural network model using provided `training_data`
        Parameters
        ----------
        training_data : 2d array_like (N, D)
            Data on which to train the tSNE model
        training_betas : 2d array_like (N,P), optional
            Widths for gaussian kernels. If `None` (the usual case), they will be calculated based on
            `training_data` and self.perplexities. One can also provide them here explicitly.
        epochs: int, optional
        verbose: int, optional
            Default 0. Verbosity level. Passed to Keras fit method

        Returns
        -------
            None. Model trained in place
        """

        assert (
            training_data.shape[1] == self.num_inputs
        ), "Input training data must be same shape as training `num_inputs`"

        self._training_betas = training_betas
        self._epochs = epochs

        if self._training_betas is None:
            training_betas = self._calc_training_betas(training_data, self.perplexities)
            self._training_betas = training_betas
        else:
            self.num_perplexities = self._get_num_perplexities(
                training_betas, self.num_perplexities
            )

        if self.do_pretrain:
            self._pretrain_layers(
                training_data,
                batch_size=self._batch_size,
                epochs=epochs,
                verbose=verbose,
            )

        self._init_loss_func()
        self.model.compile(self._optimizer, self._loss_func)

        train_generator = self._make_train_generator(
            training_data, self._training_betas, self._batch_size
        )

        batches_per_epoch = int(training_data.shape[0] // self._batch_size)

        if verbose:
            print(
                "{time}: Beginning training on {epochs} epochs".format(
                    time=datetime.datetime.now(), epochs=epochs
                )
            )
        self.model.fit(
            train_generator,
            steps_per_epoch=batches_per_epoch,
            epochs=epochs,
            verbose=verbose,
        )

        if verbose:
            print(
                "{time}: Finished training on {epochs} epochs".format(
                    time=datetime.datetime.now(), epochs=epochs
                )
            )

    def transform(self, test_data: np.ndarray) -> np.ndarray:
        """Transform the `test_data` based on trained model.
        test_data.shape[1] must match train_data.shape[1].
        That is, the test data must have the same high-dimension value.
        Parameters
        ----------
            test_data : 2d array_like (M, num_inputs)
                Data to transform using training model
        Returns
        -------
            predicted_data: 2d array_like (M, num_outputs)
        """

        assert self.model is not None, "Must train the model before transforming!"
        assert (
            test_data.shape[1] == self.num_inputs
        ), "Input test data must be same shape as training `num_inputs`"
        return self.model.predict(test_data)

    @staticmethod
    def _make_train_generator(
        training_data: np.ndarray, betas: np.ndarray, batch_size: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generator to make batches of training data. Cycles forever
        Parameters
        ----------
        training_data : 2d array_like (N, D)
        betas : 2d array_like (N, P)
        batch_size: int

        Yields
        -------
        cur_dat : 2d array_like (batch_size, D)
            Slice of `training_data`
        P_array : 2d array_like (batch_size, batch_size)
            Probability matrix between points
            This is what we use as our "true" value in the KL loss function
        """
        num_steps = training_data.shape[0] // batch_size
        cur_step = -1
        while True:
            cur_step = (cur_step + 1) % num_steps
            cur_bounds = batch_size * cur_step, batch_size * (cur_step + 1)
            cur_range = np.arange(cur_bounds[0], cur_bounds[1])
            cur_dat = training_data[cur_range, :]
            cur_betas = betas[cur_range, :]

            P_arrays_3d = _make_P_np(cur_dat, cur_betas)

            P_arrays = [P_arrays_3d[:, :, pp] for pp in range(P_arrays_3d.shape[2])]

            # Stack them along dimension 1. This is a hack
            P_arrays = np.concatenate(P_arrays, axis=1)

            yield cur_dat, P_arrays

    def save_model(self, model_path: str):
        """Save the underlying model to `model_path` using Keras"""
        return self.model.save(model_path)

    def restore_model(
        self,
        model_path: str,
        training_betas: Optional[np.ndarray] = None,
        num_perplexities: Optional[int] = None,
    ) -> None:
        """Restore the underlying model from `model_path`"""
        if not self._loss_func:
            # Have to initialize this to load the model
            self.num_perplexities = self._get_num_perplexities(
                training_betas, num_perplexities
            )
            self._init_loss_func()
        cust_objects = {self._loss_func.__name__: self._loss_func}
        self.model = tfk.models.load_model(model_path, custom_objects=cust_objects)
        self._all_layers = self.model.layers

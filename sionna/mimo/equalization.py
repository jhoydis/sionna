#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Classes and functions related to MIMO channel equalization"""

import tensorflow as tf
import sionna as sn
from sionna.mapping import Constellation, LLRs2SymbolLogits, SymbolLogits2Moments
from sionna.utils import expand_to_rank, matrix_inv, matrix_pinv, insert_dims
from sionna.mimo.utils import whiten_channel, complex2real_channel, complex2real_vector, complex2real_matrix, \
    real2complex_vector
from tensorflow.keras.layers import Layer

def lmmse_equalizer(y, h, s, whiten_interference=True):
    # pylint: disable=line-too-long
    r"""MIMO LMMSE Equalizer

    This function implements LMMSE equalization for a MIMO link, assuming the
    following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}`,
    :math:`\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`.

    The estimated symbol vector :math:`\hat{\mathbf{x}}\in\mathbb{C}^K` is given as
    (Lemma B.19) [BHS2017]_ :

    .. math::

        \hat{\mathbf{x}} = \mathop{\text{diag}}\left(\mathbf{G}\mathbf{H}\right)^{-1}\mathbf{G}\mathbf{y}

    where

    .. math::

        \mathbf{G} = \mathbf{H}^{\mathsf{H}} \left(\mathbf{H}\mathbf{H}^{\mathsf{H}} + \mathbf{S}\right)^{-1}.

    This leads to the post-equalized per-symbol model:

    .. math::

        \hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1

    where the variances :math:`\sigma^2_k` of the effective residual noise
    terms :math:`e_k` are given by the diagonal elements of

    .. math::

        \mathop{\text{diag}}\left(\mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]\right)
        = \mathop{\text{diag}}\left(\mathbf{G}\mathbf{H} \right)^{-1} - \mathbf{I}.

    Note that the scaling by :math:`\mathop{\text{diag}}\left(\mathbf{G}\mathbf{H}\right)^{-1}`
    is important for the :class:`~sionna.mapping.Demapper` although it does
    not change the signal-to-noise ratio.

    The function returns :math:`\hat{\mathbf{x}}` and
    :math:`\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}`.

    Input
    -----
    y : [...,M], tf.complex
        1+D tensor containing the received signals.

    h : [...,M,K], tf.complex
        2+D tensor containing the channel matrices.

    s : [...,M,M], tf.complex
        2+D tensor containing the noise covariance matrices.

    whiten_interference : bool
        If `True` (default), the interference is first whitened before equalization.
        In this case, an alternative expression for the receive filter is used that
        can be numerically more stable. Defaults to `True`.

    Output
    ------
    x_hat : [...,K], tf.complex
        1+D tensor representing the estimated symbol vectors.

    no_eff : tf.float
        Tensor of the same shape as ``x_hat`` containing the effective noise
        variance estimates.

    Note
    ----
    If you want to use this function in Graph mode with XLA, i.e., within
    a function that is decorated with ``@tf.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """

    # We assume the model:
    # y = Hx + n, where E[nn']=S.
    # E[x]=E[n]=0
    #
    # The LMMSE estimate of x is given as:
    # x_hat = diag(GH)^(-1)Gy
    # with G=H'(HH'+S)^(-1).
    #
    # This leads us to the per-symbol model;
    #
    # x_hat_k = x_k + e_k
    #
    # The elements of the residual noise vector e have variance:
    # diag(E[ee']) = diag(GH)^(-1) - I
    if not whiten_interference:
        # Compute G
        g = tf.matmul(h, h, adjoint_b=True) + s
        g = tf.matmul(h, matrix_inv(g), adjoint_a=True)

    else:
        # Whiten channel
        y, h  = whiten_channel(y, h, s, return_s=False) # pylint: disable=unbalanced-tuple-unpacking

        # Compute G
        i = expand_to_rank(tf.eye(tf.shape(h)[-1], dtype=s.dtype), tf.rank(s), 0)
        g = tf.matmul(h, h, adjoint_a=True) + i
        g = tf.matmul(matrix_inv(g), h, adjoint_b=True)

    # Compute Gy
    y = tf.expand_dims(y, -1)
    gy = tf.squeeze(tf.matmul(g, y), axis=-1)

    # Compute GH
    gh = tf.matmul(g, h)

    # Compute diag(GH)
    d = tf.linalg.diag_part(gh)

    # Compute x_hat
    x_hat = gy/d

    # Compute residual error variance
    one = tf.cast(1, dtype=d.dtype)
    no_eff = tf.math.real(one/d - one)

    return x_hat, no_eff

def zf_equalizer(y, h, s):
    # pylint: disable=line-too-long
    r"""MIMO ZF Equalizer

    This function implements zero-forcing (ZF) equalization for a MIMO link, assuming the
    following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}`,
    :math:`\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`.

    The estimated symbol vector :math:`\hat{\mathbf{x}}\in\mathbb{C}^K` is given as
    (Eq. 4.10) [BHS2017]_ :

    .. math::

        \hat{\mathbf{x}} = \mathbf{G}\mathbf{y}

    where

    .. math::

        \mathbf{G} = \left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}\mathbf{H}^{\mathsf{H}}.

    This leads to the post-equalized per-symbol model:

    .. math::

        \hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1

    where the variances :math:`\sigma^2_k` of the effective residual noise
    terms :math:`e_k` are given by the diagonal elements of the matrix

    .. math::

        \mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]
        = \mathbf{G}\mathbf{S}\mathbf{G}^{\mathsf{H}}.

    The function returns :math:`\hat{\mathbf{x}}` and
    :math:`\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}`.

    Input
    -----
    y : [...,M], tf.complex
        1+D tensor containing the received signals.

    h : [...,M,K], tf.complex
        2+D tensor containing the channel matrices.

    s : [...,M,M], tf.complex
        2+D tensor containing the noise covariance matrices.

    Output
    ------
    x_hat : [...,K], tf.complex
        1+D tensor representing the estimated symbol vectors.

    no_eff : tf.float
        Tensor of the same shape as ``x_hat`` containing the effective noise
        variance estimates.

    Note
    ----
    If you want to use this function in Graph mode with XLA, i.e., within
    a function that is decorated with ``@tf.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """

    # We assume the model:
    # y = Hx + n, where E[nn']=S.
    # E[x]=E[n]=0
    #
    # The ZF estimate of x is given as:
    # x_hat = Gy
    # with G=(H'H')^(-1)H'.
    #
    # This leads us to the per-symbol model;
    #
    # x_hat_k = x_k + e_k
    #
    # The elements of the residual noise vector e have variance:
    # E[ee'] = GSG'

    # Compute G
    g = matrix_pinv(h)

    # Compute x_hat
    y = tf.expand_dims(y, -1)
    x_hat = tf.squeeze(tf.matmul(g, y), axis=-1)

    # Compute residual error variance
    gsg = tf.matmul(tf.matmul(g, s), g, adjoint_b=True)
    no_eff = tf.math.real(tf.linalg.diag_part(gsg))

    return x_hat, no_eff

def mf_equalizer(y, h, s):
    # pylint: disable=line-too-long
    r"""MIMO MF Equalizer

    This function implements matched filter (MF) equalization for a
    MIMO link, assuming the following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}`,
    :math:`\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`.

    The estimated symbol vector :math:`\hat{\mathbf{x}}\in\mathbb{C}^K` is given as
    (Eq. 4.11) [BHS2017]_ :

    .. math::

        \hat{\mathbf{x}} = \mathbf{G}\mathbf{y}

    where

    .. math::

        \mathbf{G} = \mathop{\text{diag}}\left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}\mathbf{H}^{\mathsf{H}}.

    This leads to the post-equalized per-symbol model:

    .. math::

        \hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1

    where the variances :math:`\sigma^2_k` of the effective residual noise
    terms :math:`e_k` are given by the diagonal elements of the matrix

    .. math::

        \mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]
        = \left(\mathbf{I}-\mathbf{G}\mathbf{H} \right)\left(\mathbf{I}-\mathbf{G}\mathbf{H} \right)^{\mathsf{H}} + \mathbf{G}\mathbf{S}\mathbf{G}^{\mathsf{H}}.

    Note that the scaling by :math:`\mathop{\text{diag}}\left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}`
    in the definition of :math:`\mathbf{G}`
    is important for the :class:`~sionna.mapping.Demapper` although it does
    not change the signal-to-noise ratio.

    The function returns :math:`\hat{\mathbf{x}}` and
    :math:`\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}`.

    Input
    -----
    y : [...,M], tf.complex
        1+D tensor containing the received signals.

    h : [...,M,K], tf.complex
        2+D tensor containing the channel matrices.

    s : [...,M,M], tf.complex
        2+D tensor containing the noise covariance matrices.

    Output
    ------
    x_hat : [...,K], tf.complex
        1+D tensor representing the estimated symbol vectors.

    no_eff : tf.float
        Tensor of the same shape as ``x_hat`` containing the effective noise
        variance estimates.
    """

    # We assume the model:
    # y = Hx + n, where E[nn']=S.
    # E[x]=E[n]=0
    #
    # The MF estimate of x is given as:
    # x_hat = Gy
    # with G=diag(H'H)^-1 H'.
    #
    # This leads us to the per-symbol model;
    #
    # x_hat_k = x_k + e_k
    #
    # The elements of the residual noise vector e have variance:
    # E[ee'] = (I-GH)(I-GH)' + GSG'

    # Compute G
    hth = tf.matmul(h, h, adjoint_a=True)
    d = tf.linalg.diag(tf.cast(1, h.dtype)/tf.linalg.diag_part(hth))
    g = tf.matmul(d, h, adjoint_b=True)

    # Compute x_hat
    y = tf.expand_dims(y, -1)
    x_hat = tf.squeeze(tf.matmul(g, y), axis=-1)

    # Compute residual error variance
    gsg = tf.matmul(tf.matmul(g, s), g, adjoint_b=True)
    gh = tf.matmul(g, h)
    i = expand_to_rank(tf.eye(tf.shape(gsg)[-2], dtype=gsg.dtype), tf.rank(gsg), 0)

    no_eff = tf.abs(tf.linalg.diag_part(tf.matmul(i-gh, i-gh, adjoint_b=True) + gsg))
    return x_hat, no_eff

"""
This layer implements the soft-input soft-output minimum mean squared error (MMSE) parallel interference cancellation 
equalizer, as proposed in [CST2011]_. However, this modular implementation does not implement the LLR demapping, but
enables iterating the MMSE PIC equalizer on the post-equalization estimate level.

The full SISO MMSE PIC DETECTOR from [CST2011]_ computes soft (LLR) outputs
--> apply the DemapperWithPrior on the equalizer output
--> take into account that SISO MMSE PIC usually outputs EXTRINSIC LLRs to the decoder, in an IDD framework, i.e., 
subtract the prior from the LLRs. Refer to Simple_MIMO_Example.py for a code example.

[CST2011]_ C. Studer, S. Fateh, and D. Seethaler, "ASIC Implementation of Soft-Input Soft-Output
MIMO Detection Using MMSE Parallel Interference Cancellation," IEEE Journal of Solid-State Circuits,
vol. 46, no. 7, pp. 1754–1765, July 2011. https://ieeexplore.ieee.org/document/5779722
"""

class SiSoMmsePicEqualizer(Layer):
    def __init__(self,
                 num_iter,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 dtype=tf.complex64,
                 epsilon = 1e-4,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)

        assert dtype in [tf.complex64, tf.complex128], \
            "dtype must be tf.complex64 or tf.complex128"

        self._num_iter = num_iter

        # Create constellation object
        self._constellation = Constellation.create_or_check_constellation(
            constellation_type,
            num_bits_per_symbol,
            constellation,
            dtype=dtype)

        self._epsilon = epsilon
        self._realdtype = dtype.real_dtype

        self._llr2symbolLogits = LLRs2SymbolLogits(self._constellation.num_bits_per_symbol, dtype=self._realdtype)
        self._symbolLogits2moments = SymbolLogits2Moments(constellation=self._constellation, dtype=self._realdtype)

        # XLA Compatibility: XLA can't invert complex matrices, but the numerically stable MMSE PIC inverts a non
        # hermitian matrix --> with XLA, we calculate the MMSE filter in the real valued domain (self._rv_domain == True)
        self._rv_domain = sn.config.xla_compat


    def call(self, inputs):
        y, h, prior, s = inputs
        # y is unwhitened receive signal [..., M]
        # h the channel estimate [..., M, K]
        # prior the soft input LLRs [..., K, num_bits_per_symbol]
        # s the noise covariance matrix [..., M, M]

        ## preprocessing
        # Whiten channel
        y, h = whiten_channel(y, h, s, return_s=False)  # pylint: disable=unbalanced-tuple-unpacking

        # matched filtering of y
        y_mf = tf.expand_dims(tf.linalg.matvec(h, y, adjoint_a=True), -1)      # y_mf is [..., K, 1]

        ## Step 1: compute Gramm matrix
        g = tf.matmul(h, h, adjoint_a=True)     # g is [..., K, K]

        # real valued gram matrix
        if self._rv_domain:
            hr = complex2real_matrix(h)     # hr is [..., 2M, 2K]
            gr = tf.matmul(hr, hr, adjoint_a=True)      # gr is [..., 2K, 2K]

        # Step 2: compute soft symbol estimates and variances
        x_hat, var_x = self._symbolLogits2moments(self._llr2symbolLogits(prior))      # both are [..., K]

        def mmse_pic_iteration(x_hat, var_x, it):
            # Step 3: perform parallel interference cancellation
            # H^H y_hat_i = y_mf - sum_j!=i gj x_hat_j = y + g_i x_hat_i - sum_j g_j x_hat_j
            y_mf_pic = y_mf + g * insert_dims(x_hat, num_dims=1, axis=-2) \
                       - tf.linalg.matmul(g, insert_dims(x_hat, num_dims=1, axis=-1))
            # y_mf_pic is [..., K, K]

            # Step 4: compute A^-1 matrix
            # Calculate MMSE Filter (efficiently)
            # W^H = A^-1 H^H
            # A = H^H H \Lambda + N_0 I_Mt
            # \Lambda_ii is a diagonal matrix with \Lambda_ii = E_i = error_var

            if self._rv_domain:
                # stack error variances and make it real (imaginary part is zero anyway)
                var_x = tf.cast(tf.concat([var_x, var_x], axis=-1), dtype=self._realdtype)
                var_x_row_vec = insert_dims(var_x, num_dims=1, axis=-2)
                a = gr * var_x_row_vec
                # a is [..., 2K, 2K]
            else:
                var_x_row_vec = tf.cast(insert_dims(var_x, num_dims=1, axis=-2), self.dtype)
                a = g * var_x_row_vec
                # a is [..., K, K]

            i = expand_to_rank(tf.eye(tf.shape(a)[-1], dtype=a.dtype), tf.rank(a), 0)
            a = a + i

            a_inv = tf.linalg.inv(a)    # a is non-hermitian! that's why we can't use sn.utils.matrix_inv

            # Step 5: compute unbiased MMSE filter and outputs, calculate A\H^H

            # calculate bias mu_i = diag(A^-1 H^H H) = diag(A^-1 G)
            # diagonal of matrix matrix multiplication simplified to sum and dot-product
            if self._rv_domain:
                mu = tf.reduce_sum(a_inv * tf.linalg.matrix_transpose(gr), axis=-1)
                # mu is [..., 2K]
            else:
                mu = tf.math.real(tf.reduce_sum(a_inv * tf.linalg.matrix_transpose(g), axis=-1))
                # mu is [..., K]

            y_mf_pic_trans = tf.linalg.matrix_transpose(y_mf_pic)
            if self._rv_domain:
                # make y_mf_pic columns real (after transposition, the last dimension corresponds to vectors)
                y_mf_pic_trans = complex2real_vector(y_mf_pic_trans)        # is [..., K, 2K]
                # stack them such to make y_mf_pic_trans [..., 2K, 2K]
                y_mf_pic_trans = tf.concat([y_mf_pic_trans, y_mf_pic_trans], axis=-2)

            # efficient parallel equalization after PIC (z_i = i'th row of a_inv * y_MF_PIC_i)
            # boils down to tf.reduce_sum(a_inv * y_mf_pic_trans, axis=-1)
            # divide by mu_i for unbiasedness
            x_hat = tf.reduce_sum(a_inv * y_mf_pic_trans, axis=-1) / tf.cast(mu, dtype=a_inv.dtype)  # is [..., K] or [..., 2K]

            # compute post equalization signal error estimate: rho_i = mu_i / (1 - var_x_i * mu_i)
            # 1 - var_x_i * mu_i can become numerically 0 (or even slightly smaller than zero due to limited numerical precision)
            var_x = tf.divide(mu, tf.maximum(1 - var_x * mu, self._epsilon)) # is [..., K] or [..., 2K]

            if self._rv_domain:
                x_hat = real2complex_vector(x_hat)
                var_x, _ = tf.split(var_x, 2, -1)   # real variances map to the same complex valued variances in this model

            return x_hat, var_x, it

        # stopping condition (required for tf.while_loop)
        def dec_stop(x_hat, var_x, it):  # pylint: disable=W0613
            return tf.less(it, self._num_iter)

        # start decoding iterations
        it = tf.constant(0)
        x_hat, var_x, _ = tf.while_loop(dec_stop,
                                     mmse_pic_iteration,
                                     (x_hat, var_x, it),
                                     parallel_iterations=1,
                                     maximum_iterations=self._num_iter)

        no_eff = 1/var_x
        return x_hat, no_eff

#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition and functions related to OFDM channel equalization"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import sionna
from sionna.utils import flatten_dims, split_dim, flatten_last_dims, expand_to_rank, selectDataCarryingOFDMSymbols, \
    matrix_sqrt_inv
from sionna.mimo import lmmse_equalizer
from sionna.ofdm import RemoveNulledSubcarriers
import numpy as np


class LMMSEEqualizer(Layer):
    # pylint: disable=line-too-long
    """LMMSEEqualizer(resource_grid, stream_management, whiten_interference=True, dtype=tf.complex64, **kwargs)

    LMMSE equalization for OFDM MIMO transmissions.

    This layer computes linear minimum mean squared error (LMMSE) estimation
    for OFDM MIMO transmissions. The OFDM and stream configuration are provided
    by a :class:`~sionna.ofdm.ResourceGrid` and
    :class:`~sionna.mimo.StreamManagement` instance, respectively. The
    detection algorithm is the :meth:`~sionna.mimo.lmmse_equalizer`. The layer
    computes soft-symbol estimates together with effective noise variances
    for all streams which can, e.g., be used by a
    :class:`~sionna.mapping.Demapper` to obtain LLRs.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    stream_management : StreamManagement
        An instance of :class:`~sionna.mimo.StreamManagement`.

    whiten_interference : bool
        If `True` (default), the interference is first whitened before equalization.
        In this case, an alternative expression for the receive filter is used which
        can be numerically more stable.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    (y, h_hat, err_var, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        The received OFDM resource grid after cyclic prefix removal and FFT.

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex
        The channel estimates for all streams from all transmitters.

    err_var : [Broadcastable to shape of ``h_hat``], tf.float
        The variance of the channel estimation error.

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float
        The variance of the AWGN noise.

    Output
    ------
    x_hat : [batch_size, num_tx, num_streams, num_data_symbols], tf.complex
        The estimated symbols.

    no_eff : [batch_size, num_tx, num_streams, num_data_symbols], tf.float
        The effective noise variance for each estimated symbol.

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@tf.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 whiten_interference=True,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert isinstance(resource_grid, sionna.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.mimo.StreamManagement)
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._whiten_interference = whiten_interference
        self._removed_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

        # Precompute indices to extract data symbols
        mask = resource_grid.pilot_pattern.mask
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        data_ind = tf.argsort(flatten_last_dims(mask), direction="ASCENDING")
        self._data_ind = data_ind[...,:num_data_symbols]

    def call(self, inputs):

        y, h_hat, err_var, no = inputs
        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]

        # h_hat has shape:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]

        # err_var has a shape that is broadcastable to h_hat

        # no has shape [batch_size, num_rx, num_rx_ant]
        # or just the first n dimensions of this

        # Remove nulled subcarriers from y (guards, dc). New shape:
        # [batch_size, num_rx, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        y_eff = self._removed_nulled_scs(y)

        ####################################################
        ### Prepare the observation y for MIMO detection ###
        ####################################################
        # Transpose y_eff to put num_rx_ant last. New shape:
        # [batch_size, num_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, num_rx_ant]
        y_dt = tf.transpose(y_eff, [0, 1, 3, 4, 2])
        y_dt = tf.cast(y_dt, self._dtype)

        ##############################################
        ### Prepare the err_var for MIMO detection ###
        ##############################################
        # New shape is:
        # [batch_size, num_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, num_rx_ant, num_tx*num_streams]
        err_var_dt = tf.broadcast_to(err_var, tf.shape(h_hat))
        err_var_dt = tf.transpose(err_var_dt, [0, 1, 5, 6, 2, 3, 4])
        err_var_dt = flatten_last_dims(err_var_dt, 2)
        err_var_dt = tf.cast(err_var_dt, self._dtype)

        ###############################
        ### Construct MIMO channels ###
        ###############################

        # Reshape h_hat for the construction of desired/interfering channels:
        # [num_rx, num_tx, num_streams_per_tx, batch_size, num_rx_ant, ,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        perm = [1, 3, 4, 0, 2, 5, 6]
        h_dt = tf.transpose(h_hat, perm)

        # Flatten first tthree dimensions:
        # [num_rx*num_tx*num_streams_per_tx, batch_size, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        h_dt = flatten_dims(h_dt, 3, 0)

        # Gather desired and undesired channels
        ind_desired = self._stream_management.detection_desired_ind
        ind_undesired = self._stream_management.detection_undesired_ind
        h_dt_desired = tf.gather(h_dt, ind_desired, axis=0)
        h_dt_undesired = tf.gather(h_dt, ind_undesired, axis=0)

        # Split first dimension to separate RX and TX:
        # [num_rx, num_streams_per_rx, batch_size, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        h_dt_desired = split_dim(h_dt_desired, [self._stream_management.num_rx, -1], 0)
        h_dt_undesired = split_dim(h_dt_undesired, [self._stream_management.num_rx, -1], 0)

        # Permutate dims to
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,..
        #  ..., num_rx_ant, num_streams_per_rx(num_Interfering_streams_per_rx)]
        perm = [2, 0, 4, 5, 3, 1]
        h_dt_desired = tf.transpose(h_dt_desired, perm)
        h_dt_desired = tf.cast(h_dt_desired, self._dtype)
        h_dt_undesired = tf.transpose(h_dt_undesired, perm)

        ##################################
        ### Prepare the noise variance ###
        ##################################
        # no is first broadcast to [batch_size, num_rx, num_rx_ant]
        # then the rank is expanded to that of y
        # then it is transposed like y to the final shape
        # [batch_size, num_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, num_rx_ant]
        no_dt = expand_to_rank(no, 3, -1)
        no_dt = tf.broadcast_to(no_dt, tf.shape(y)[:3])
        no_dt = expand_to_rank(no_dt, tf.rank(y), -1)
        no_dt = tf.transpose(no_dt, [0,1,3,4,2])
        no_dt = tf.cast(no_dt, self._dtype)

        ##################################################
        ### Compute the interference covariance matrix ###
        ##################################################
        # Covariance of undesired transmitters
        s_inf = tf.matmul(h_dt_undesired, h_dt_undesired, adjoint_b=True)

        #Thermal noise
        s_no = tf.linalg.diag(no_dt)

        # Channel estimation errors
        # As we have only error variance information for each element,
        # we simply sum them across transmitters and build a
        # diagonal covariance matrix from this
        s_csi = tf.linalg.diag(tf.reduce_sum(err_var_dt, -1))

        # Final covariance matrix
        s = s_inf + s_no + s_csi
        s = tf.cast(s, self._dtype)

        ############################################################
        #### Compute LMMSE estimate and effective noise variance ###
        ############################################################
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,...
        #  ..., num_stream_per_rx]
        x_hat, no_eff = lmmse_equalizer(y_dt, h_dt_desired,
                                        s, self._whiten_interference)

        ################################################
        ### Extract data symbols for all detected TX ###
        ################################################
        # Transpose tensor to shape
        # [num_rx, num_streams_per_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, batch_size]
        x_hat = tf.transpose(x_hat, [1, 4, 2, 3, 0])
        no_eff = tf.transpose(no_eff, [1, 4, 2, 3, 0])

        # Merge num_rx amd num_streams_per_rx
        # [num_rx * num_streams_per_rx, num_ofdm_symbols,...
        #  ...,num_effective_subcarriers, batch_size]
        x_hat = flatten_dims(x_hat, 2, 0)
        no_eff = flatten_dims(no_eff, 2, 0)

        # Put first dimension into the right ordering
        stream_ind = self._stream_management.stream_ind
        x_hat = tf.gather(x_hat, stream_ind, axis=0)
        no_eff = tf.gather(no_eff, stream_ind, axis=0)

        # Reshape first dimensions to [num_tx, num_streams] so that
        # we can compared to the way the streams were created.
        # [num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers,...
        #  ..., batch_size]
        num_streams = self._stream_management.num_streams_per_tx
        num_tx = self._stream_management.num_tx
        x_hat = split_dim(x_hat, [num_tx, num_streams], 0)
        no_eff = split_dim(no_eff, [num_tx, num_streams], 0)

        # Flatten resource grid dimensions
        # [num_tx, num_streams, num_ofdm_symbols*num_effective_subcarriers,...
        #  ..., batch_size]
        x_hat = flatten_dims(x_hat, 2, 2)
        no_eff = flatten_dims(no_eff, 2, 2)

        # Broadcast no_eff to the shape of x_hat
        no_eff = tf.broadcast_to(no_eff, tf.shape(x_hat))

        # Gather data symbols
        # [num_tx, num_streams, num_data_symbols, batch_size]
        x_hat = tf.gather(x_hat, self._data_ind, batch_dims=2, axis=2)
        no_eff = tf.gather(no_eff, self._data_ind, batch_dims=2, axis=2)

        # Put batch_dim first
        # [batch_size, num_tx, num_streams, num_data_symbols]
        x_hat = tf.transpose(x_hat, [3, 0, 1, 2])
        no_eff = tf.transpose(no_eff, [3, 0, 1, 2])

        return (x_hat, no_eff)

class SisoMmsePicDetector(Layer):
    # pylint: disable=line-too-long
    """SisoMmsePicDetector(resource_grid,
                 stream_management,
                 demapping_method,
                 constellation: sionna.mapping.Constellation,
                 dtype=tf.complex64, low_complexity=False,
                 regularizationEpsilon=1e-4, data_carrying_whitened_inputs=False,
                 **kwargs)

        MMSE PIC equalization for OFDM MIMO transmissions.

        This layer computes the soft-input soft-output minimum mean squared error (MMSE) parallel interference cancellation detector
        for OFDM MIMO transmissions, as proposed in [CST2011]_. The OFDM and stream configuration are provided
        by a :class:`~sionna.ofdm.ResourceGrid` and
        :class:`~sionna.mimo.StreamManagement` instance, respectively. The layer
        computes log-likelihood ratio (LLR) estimates for each bit.

        Current implementation only works with single-cell, only tested with single-antenna UEs

        [CST2011]_ C. Studer, S. Fateh, and D. Seethaler, "ASIC Implementation of Soft-Input Soft-Output
        MIMO Detection Using MMSE Parallel Interference Cancellation," IEEE Journal of Solid-State Circuits,
        vol. 46, no. 7, pp. 1754–1765, July 2011. https://ieeexplore.ieee.org/document/5779722

        Parameters
        ----------
        resource_grid : ResourceGrid
            An instance of :class:`~sionna.ofdm.ResourceGrid`.

        stream_management : StreamManagement
            An instance of :class:`~sionna.mimo.StreamManagement`.

        demapping_method : One of ["app", "maxlog"], str
        The method used for computing the LLRs.

        constellatoin : Constellation
            An instance of :class:`~sionna.mapping.Constellation`

        dtype : tf.Dtype
            Datatype for internal calculations and the output dtype.
            Defaults to `tf.complex64`.

        low_complexity : selects if low_complexity LLR-to-symbol mapping and demapping is applied (defaults to False)
            Refer to https://ieeexplore.ieee.org/abstract/document/4025128 and Table I, II, III, IV in https://ieeexplore.ieee.org/document/984761

        regularizationEpsilon : regularization constant to avoid dividing by zero (defaults to 1e-4)

        data_carrying_whitened_inputs: boolean flag selecting if input data represents noise-whitened data-carrying elements y, h, etc. (defaults to False)

        Input
        -----
        (y, h_hat, err_var, no, llr_a, G) :
            Tuple:

        y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
            The received OFDM resource grid after cyclic prefix removal and FFT.
            If data_carrying_whitened_inputs==True:  received OFDM resource grid after noise-whitening and pilot removal
            num_ofdm_data_symbols=int(self._resource_grid.num_data_symbols / num_effective_subcarriers)
            [batch_size, num_rx, num_rx_ant, num_ofdm_data_symbols, fft_size]
            If also the Gram matrix G is not None, y is the input signal AFTER matched filtering
            [batch_size, 1, num_ofdm_data_symbols, num_effective_subcarriers, num_streams_per_rx]

        h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex
            The channel estimates for all streams from all transmitters.
            If data_carrying_whitened_inputs==True:  received channels for OFDM resource grid after noise-whitening and pilot removal
            num_ofdm_data_symbols=int(self._resource_grid.num_data_symbols / num_effective_subcarriers)
            [batch_size, 1, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_data_symbols, num_effective_subcarriers]

        err_var : [Broadcastable to shape of ``h_hat``], tf.float
            The variance of the channel estimation error.

        no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float
            The variance of the AWGN noise.

        llr_a : a-priori LLRs from previous IDD iterations
            None | [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float

        G : Gram matrix corresponding to (H'*H)
            None | [batch_size, 1, num_ofdm_data_symbols, num_effective_subcarriers, num_streams_per_rx, num_streams_per_rx], tf.complex

        Output
        ------
        [llr_e, y_MF, h_dt_desired_whitened, G]
        llr_e : extrinsic LLR estimates
            [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol]

        y_MF : input data after noise-whitening, pilot removal and matched filtering
            [batch_size, 1, num_ofdm_data_symbols, num_effective_subcarriers, num_streams_per_rx]

        h_dt_desired_whitened : intra-cell channel matrices after noise whitening and pilot removal
            [batch_size, 1, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_data_symbols, num_effective_subcarriers]

        G : Gram Matrix
            None | [batch_size, 1, num_ofdm_data_symbols, num_effective_subcarriers, num_streams_per_rx, num_streams_per_rx], tf.complex

        Note
        ----
        The current implementation is limited to single cell and single antenna UEs.
        """

    def __init__(self,
                 resource_grid,
                 stream_management,
                 demapping_method,
                 constellation: sionna.mapping.Constellation,
                 dtype=tf.complex64, low_complexity=False,
                 regularizationEpsilon=1e-4, data_carrying_whitened_inputs=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert isinstance(resource_grid, sionna.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.mimo.StreamManagement)
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._removed_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)
        self._constellation = constellation
        self._dtype = dtype
        self._epsilon = regularizationEpsilon
        self._low_complexity = low_complexity
        self._data_carrying_whitened_inputs = data_carrying_whitened_inputs

        # Precompute indices to extract data symbols
        mask = resource_grid.pilot_pattern.mask
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        data_ind = tf.argsort(flatten_last_dims(mask), direction="ASCENDING")
        self._data_ind = data_ind[..., :num_data_symbols]

        # Create boolean mask for LLR to Symbol mapping
        num_bits_per_symbol = self._constellation.num_bits_per_symbol
        num_points = int(2 ** num_bits_per_symbol)
        a = np.zeros([num_points, num_bits_per_symbol])
        for i in range(0, num_points):
            a[i, :] = np.array(list(np.binary_repr(i, num_bits_per_symbol)),
                               dtype=np.int16)

        self._a = a
        self._aBool = tf.cast(self._a, tf.bool)

        # Compute symbol indices for which the bits are 0 or 1
        c0 = np.zeros([int(num_points / 2), num_bits_per_symbol])
        c1 = np.zeros([int(num_points / 2), num_bits_per_symbol])
        for i in range(num_bits_per_symbol - 1, -1, -1):
            c0[:, i] = np.where(a[:, i] == 0)[0]
            c1[:, i] = np.where(a[:, i] == 1)[0]
        self._c0 = tf.constant(c0, dtype=tf.int32)  # Symbols with ith bit=0
        self._c1 = tf.constant(c1, dtype=tf.int32)  # Symbols with ith bit=1

        # normalization of constellation, important for low-complexity LLR-to-soft-symbol mapping and demapping
        if constellation.normalize:
            n = int(num_bits_per_symbol / 2)
            qam_var = 1 / (2 ** (n - 2)) * np.sum(np.linspace(1, 2 ** n - 1, 2 ** (n - 1)) ** 2)
            self._qam_normalization_factor = 1 / np.sqrt(qam_var)
        else:
            self._qam_normalization_factor = 1

        if demapping_method == "app":
            self._reduce = tf.reduce_logsumexp
        else:
            self._reduce = tf.reduce_max

        # XLA Compatibility: XLA can't invert complex matrices, numerically stable MMSE PIC matrix inverse is non
        # hermitian --> with XLA, we calculate the MMSE filter in the real valued domain (self._rv_domain == True)
        self._rv_domain = sionna.config.xla_compat

    def soft_symbols(self, llr_a, points_reshaped, batch_size, num_ofdm_symbols, num_effective_subcarriers, num_tx,
                     num_streams):

        # llr_a is [batch_size, num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers, num_bits_per_symbol]
        p0 = 0.5 * (1 - tf.math.tanh(
            0.5 * llr_a))

        if self._low_complexity and (self._constellation._constellation_type == "qam" and
                                     self._constellation.num_bits_per_symbol in [2, 4, 6] or
                                     self._constellation._constellation_type == "pam" and
                                     self._constellation.num_bits_per_symbol == 1):
            p1 = 1 - p0
            if self._constellation.num_bits_per_symbol == 1:
                # BPSK
                s_real = (1 - 2 * tf.gather(p1, indices=0, axis=-1))
                s_imag = 0

                c = 1
                d = 0
            elif self._constellation.num_bits_per_symbol == 2:
                # 4-QAM
                s_real = (1 - 2 * tf.gather(p1, indices=0, axis=-1))
                s_imag = (1 - 2 * tf.gather(p1, indices=1, axis=-1))

                c = 2
                d = 0
            elif self._constellation.num_bits_per_symbol == 4:
                # 16-QAM
                s_real = (1 - 2 * tf.gather(p1, indices=0, axis=-1)) * (1 + 2 * tf.gather(p1, indices=2, axis=-1))
                s_imag = (1 - 2 * tf.gather(p1, indices=1, axis=-1)) * (1 + 2 * tf.gather(p1, indices=3, axis=-1))

                c = 1 + 8 * tf.gather(p1, indices=2, axis=-1)
                d = 1 + 8 * tf.gather(p1, indices=3, axis=-1)
            elif self._constellation.num_bits_per_symbol == 6:
                # 64-QAM
                raise Exception('constellation order not implemented')
            else:
                raise Exception('unsupported constellation order')

            s_hat = self._qam_normalization_factor * tf.complex(s_real, s_imag)
            error_var = self._qam_normalization_factor ** 2 * ((c + d) - tf.square(s_real) - tf.square(s_imag))

            log_P_C = None
        else:
            p0 = tf.expand_dims(p0, axis=-2)
            p1 = 1 - p0
            oneBits_reshaped = tf.reshape(self._aBool, [1, 1, 1, 1, 1] + self._constellation.points.shape +
                                          self._constellation.num_bits_per_symbol)
            pC_bits = tf.where(oneBits_reshaped, p1, p0)

            # probability of each constellation symbol
            P_C = tf.reduce_prod(pC_bits, axis=-1)

            # numerically stable way to calculate log_pC (log of constellation symbol probabilities)
            # following (22), (23) from C. Studer, "Soft–Input Soft–Output Single Tree-Search
            # Sphere Decoding," IEEE TRANS. ON INFORMATION THEORY, VOL. 56, NO. 10, OCTOBER 2010
            abs_llrs = tf.math.abs(llr_a)
            K_i_tilde = tf.reduce_sum(0.5 * abs_llrs + tf.math.log(1 + tf.math.exp(-abs_llrs)), axis=-1,
                                      keepdims=True)  # @TODO: check axis right?

            x_ib = 2 * (tf.cast(oneBits_reshaped, dtype=tf.float32) - 0.5)
            log_P_C = - (K_i_tilde - tf.reduce_sum(0.5 * x_ib * tf.expand_dims(llr_a, axis=-2), axis=-1))

            # s_hat [batch_size, num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers]
            s_hat = tf.reduce_sum(points_reshaped * tf.cast(P_C, tf.complex64), axis=-1)

            # Calculate Error Variance Estimate
            squared_error = tf.math.pow(
                tf.maximum(tf.abs(tf.expand_dims(s_hat, axis=-1) - points_reshaped), self._epsilon), 2)
            error_var = tf.reduce_sum(squared_error * P_C, axis=-1)

        # transform s_hat and error_var to [batch_size, 1, num_ofdm_symbols, num_effective_subcarriers, ...
        # num_tx*num_streams, 1]
        s_hat = tf.transpose(s_hat, [0, 3, 4, 1, 2])
        error_var = tf.transpose(error_var, [0, 3, 4, 1, 2])
        s_int_shape = tf.concat(
            [[batch_size], [1], [num_ofdm_symbols], [num_effective_subcarriers], [num_tx * num_streams, 1]], 0)
        s_hat = tf.reshape(s_hat, s_int_shape)
        error_var = tf.reshape(error_var, s_int_shape)

        return [s_hat, error_var, log_P_C]

    def LLRCalculation(self, z_i, rho_i, points_reshaped, log_P_C):
        if self._low_complexity and (self._constellation._constellation_type == "qam" and
                                     self._constellation.num_bits_per_symbol in [2, 4, 6] or
                                     self._constellation._constellation_type == "pam" and
                                     self._constellation.num_bits_per_symbol == 1):
            # transform z_i to constellation w/o unit-energy scaling
            z_i = z_i / self._qam_normalization_factor

            if self._constellation.num_bits_per_symbol == 1:
                # BPSK
                lambda_b_1 = 4 * tf.math.real(z_i)
                lambda_b = lambda_b_1
            elif self._constellation.num_bits_per_symbol == 2:
                # 4-QAM
                lambda_b_1 = 4 * tf.math.real(z_i)
                lambda_b_2 = 4 * tf.math.imag(z_i)
                lambda_b = tf.concat([lambda_b_1, lambda_b_2], axis=-1)
            elif self._constellation.num_bits_per_symbol == 4:
                # 16-QAM
                z_i_real = tf.math.real(z_i)
                z_i_imag = tf.math.imag(z_i)
                lambda_b_1 = tf.where(tf.math.less_equal(tf.abs(z_i_real), 2), 4 * z_i_real,
                                      8 * z_i_real - 8 * tf.sign(z_i_real))
                lambda_b_2 = 8 - 4 * tf.abs(z_i_real)
                lambda_b_3 = tf.where(tf.math.less_equal(tf.abs(z_i_imag), 2), 4 * z_i_imag,
                                      8 * z_i_imag - 8 * tf.sign(z_i_imag))
                lambda_b_4 = 8 - 4 * tf.abs(z_i_imag)
                lambda_b = tf.concat([lambda_b_1, lambda_b_3, lambda_b_2, lambda_b_4], axis=-1)
            elif self._constellation.num_bits_per_symbol == 6:
                # 64-QAM
                raise Exception('constellation order not implemented')
            else:
                raise Exception('unsupported constellation order')

            lambda_b = self._qam_normalization_factor ** 2 * lambda_b
            llr_d = - rho_i * lambda_b  # minus because of inverse LLR definition
        else:
            squared_dist = tf.math.pow(tf.math.abs(z_i - points_reshaped), 2)

            squared_dist = tf.maximum(squared_dist, self._epsilon ** 2)  #

            if log_P_C is not None:
                exponents = -squared_dist * rho_i + log_P_C  # intrinsic
            else:
                exponents = -squared_dist * rho_i  # extrinsic

            exp0 = tf.gather(exponents, self._c0, axis=-1, batch_dims=0)
            exp1 = tf.gather(exponents, self._c1, axis=-1, batch_dims=0)

            # transform
            # llr_d is [batch_size, 1, num_ofdm_symbols, num_effective_subcarriers, num_tx*num_streams, ...
            # num_bits_per_symbol]
            llr_d = self._reduce(exp1, axis=-2) - self._reduce(exp0, axis=-2)  # max log?? or app??

        return llr_d

    def call(self, inputs):
        y, h_hat, err_var, no, llr_a, G = inputs

        # prepare variables for shape
        batch_size = tf.shape(y)[0]
        num_effective_subcarriers = self._resource_grid.num_effective_subcarriers
        num_ofdm_data_symbols = int(self._resource_grid.num_data_symbols / num_effective_subcarriers)
        num_bits_per_symbol = self._constellation.num_bits_per_symbol
        num_tx = self._resource_grid.num_tx
        num_points = int(self._constellation.points.shape[0])
        num_streams = self._resource_grid.num_streams_per_tx
        num_data_symbols = int(self._resource_grid.num_data_symbols)
        _type_float = tf.float32
        data_ind = self._data_ind[0, 0, :]

        if not self._data_carrying_whitened_inputs:
            # Remove nulled subcarriers from y (guards, dc). New shape:
            # [batch_size, num_rx, num_rx_ant, ...
            #  ..., num_ofdm_symbols, num_effective_subcarriers]
            y_eff = self._removed_nulled_scs(y)
            ####################################################
            ### Prepare the observation y for MIMO detection ###
            ####################################################
            # Transpose y_eff to put num_rx_ant last. New shape:
            # [batch_size, num_rx, num_ofdm_symbols,...
            #  ..., num_effective_subcarriers, num_rx_ant]
            y_dt = tf.transpose(y_eff, [0, 1, 3, 4, 2])
            y_dt = tf.cast(y_dt, self._dtype)

            # Gather only data-carrying symbols
            # New shape:
            # [batch_size, num_rx, num_ofdm_data_symbols,...
            #  ..., num_effective_subcarriers, num_rx_ant]
            y_dt = selectDataCarryingOFDMSymbols(y_dt, 2, data_ind, num_ofdm_data_symbols, num_effective_subcarriers)

            ##############################################
            ### Prepare the err_var for MIMO detection ###
            ##############################################
            # New shape is:
            # [batch_size, num_rx, num_ofdm_data_symbols,...
            #  ..., num_effective_subcarriers, num_rx_ant, num_tx*num_streams]
            err_var_dt = tf.broadcast_to(err_var, tf.shape(h_hat))
            err_var_dt = tf.transpose(err_var_dt, [0, 1, 5, 6, 2, 3, 4])
            err_var_dt = flatten_last_dims(err_var_dt, 2)
            err_var_dt = tf.cast(err_var_dt, self._dtype)
            err_var_dt = selectDataCarryingOFDMSymbols(err_var_dt, 2, data_ind, num_ofdm_data_symbols,
                                                       num_effective_subcarriers)

            ###############################
            ### Construct MIMO channels ###
            ###############################

            # Reshape h_hat for the construction of desired/interfering channels:
            # [num_rx, num_tx, num_streams_per_tx, batch_size, num_rx_ant, ,...
            #  ..., num_ofdm_symbols, num_effective_subcarriers]
            perm = [1, 3, 4, 0, 2, 5, 6]
            h_dt = tf.transpose(h_hat, perm)

            # Flatten first three dimensions:
            # [num_rx*num_tx*num_streams_per_tx, batch_size, num_rx_ant, ...
            #  ..., num_ofdm_symbols, num_effective_subcarriers]
            h_dt = flatten_dims(h_dt, 3, 0)

            # Gather desired and undesired channels
            ind_desired = self._stream_management.detection_desired_ind
            ind_undesired = self._stream_management.detection_undesired_ind
            h_dt_desired = tf.gather(h_dt, ind_desired, axis=0)
            h_dt_undesired = tf.gather(h_dt, ind_undesired, axis=0)

            # Split first dimension to separate RX and TX:
            # [num_rx, num_streams_per_rx, batch_size, num_rx_ant, ...
            #  ..., num_ofdm_symbols, num_effective_subcarriers]
            h_dt_desired = split_dim(h_dt_desired, [self._stream_management.num_rx, -1], 0)
            h_dt_undesired = split_dim(h_dt_undesired, [self._stream_management.num_rx, -1], 0)

            # Permutate dims to
            # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,..
            #  ..., num_rx_ant, num_streams_per_rx(num_Interfering_streams_per_rx)]
            perm = [2, 0, 4, 5, 3, 1]
            h_dt_desired = tf.transpose(h_dt_desired, perm)
            h_dt_desired = tf.cast(h_dt_desired, self._dtype)
            h_dt_undesired = tf.transpose(h_dt_undesired, perm)
            h_dt_desired = selectDataCarryingOFDMSymbols(h_dt_desired, 2, data_ind, num_ofdm_data_symbols,
                                                         num_effective_subcarriers)
            h_dt_undesired = selectDataCarryingOFDMSymbols(h_dt_undesired, 2, data_ind, num_ofdm_data_symbols,
                                                           num_effective_subcarriers)

            ##################################
            ### Prepare the noise variance ###
            ##################################
            # no is first broadcast to [batch_size, num_rx, num_rx_ant]
            # then the rank is expanded to that of y
            # then it is transposed like y to the final shape
            # [batch_size, num_rx, num_ofdm_symbols,...
            #  ..., num_effective_subcarriers, num_rx_ant]
            no_dt = expand_to_rank(no, 3, -1)
            no_dt = tf.broadcast_to(no_dt, tf.shape(y)[:3])
            no_dt = expand_to_rank(no_dt, tf.rank(y), -1)
            no_dt = tf.transpose(no_dt, [0, 1, 3, 4, 2])
            no_dt = tf.cast(no_dt, self._dtype)

            ##################################################
            ### Compute the interference covariance matrix ###
            ##################################################
            # Covariance of undesired transmitters
            s_inf = tf.matmul(h_dt_undesired, h_dt_undesired, adjoint_b=True)

            # Thermal noise
            s_no = tf.linalg.diag(no_dt)

            # Channel estimation errors
            # As we have only error variance information for each element,
            # we simply sum them across transmitters and build a
            # diagonal covariance matrix from this
            s_csi = tf.linalg.diag(tf.reduce_sum(err_var_dt, -1))

            # Final covariance matrix
            s = s_inf + s_no + s_csi
            s = tf.cast(s, self._dtype)

            # Noise+Interference Whitening
            s_inv_1_2 = matrix_sqrt_inv(s)

            # Whiten the observation
            y_dt = tf.expand_dims(y_dt, -1)
            y_dt_whitened = tf.matmul(s_inv_1_2, y_dt)

            # Compute channel after whitening
            h_dt_desired_whitened = tf.matmul(s_inv_1_2, h_dt_desired)

            # Step 1: Compute Gram matrix
            # h_dt_desired is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,..
            #  ..., num_rx_ant, num_streams_per_rx(num_Interfering_streams_per_rx)]
            # G is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, ...
            # num_streams_per_rx]
        else:
            h_dt_desired_whitened = h_hat
            y_dt_whitened = y
        if G is None:
            G = tf.linalg.matmul(h_dt_desired_whitened, h_dt_desired_whitened,
                                 adjoint_a=True)

            # y_MF is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx]
            y_MF = tf.linalg.matmul(h_dt_desired_whitened, y_dt_whitened, adjoint_a=True)
        else:
            y_MF = y

        _G_rv = 0
        if self._rv_domain:
            H_R_1 = tf.concat([tf.math.real(h_dt_desired_whitened), tf.math.imag(h_dt_desired_whitened)], axis=-2)
            H_R_2 = tf.concat([-tf.math.imag(h_dt_desired_whitened), tf.math.real(h_dt_desired_whitened)], axis=-2)
            H_R = tf.concat([H_R_1, H_R_2], axis=-1)
            _G_rv = tf.matmul(H_R, H_R, adjoint_a=True)
        ############################################################
        #### SISO LMMSE PIC ###
        # following Algorithm 1 from [CST2011]_
        ############################################################

        # Calculate Soft Symbols
        points_reshaped = tf.reshape(self._constellation.points, [1] * 5 + [num_points])

        if llr_a is None:
            # no a priori LLR => no parallel interference cancellation
            y_hat_i_MF = y_MF
            # _lambda = None
            _error_var_row_vec = None
            log_P_C = None
            error_var = 1
            llr_a_out = 0
            if self._rv_domain:
                _A = _G_rv
            else:
                _A = G
        else:
            # Step 2: Calculte soft-symbols and variances

            # llr_a is [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol]
            # reshape to [batch_size, num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers, ...
            # num_bits_per_symbol]
            llr_a_out = llr_a
            llr_a = tf.expand_dims(llr_a, axis=-1)
            llr_a = tf.expand_dims(llr_a, axis=-3)
            llr_int_shape = tf.concat(
                [tf.shape(llr_a)[:-3], [num_ofdm_data_symbols, num_effective_subcarriers, num_bits_per_symbol]], 0)
            llr_a = tf.reshape(llr_a, llr_int_shape)

            [s_hat, error_var, log_P_C] = self.soft_symbols(llr_a, points_reshaped, batch_size, num_ofdm_data_symbols,
                                                            num_effective_subcarriers, num_tx, num_streams)

            # Step 3: Perform PIC
            # H^H y_hat_i = y_MF - sum_j!=i gj s_hat_j = y + g_i s_hat_i - sum_j g_j s_hat_j
            # y_MF is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx]
            # G is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, ...
            # num_streams_per_rx]
            _g_j_s_hat_j = tf.linalg.matmul(G, s_hat)
            _s_hat = tf.transpose(s_hat, [0, 1, 2, 3, 5, 4])
            y_hat_i_MF = y_MF + G * _s_hat - _g_j_s_hat_j
            # y_hat_i_MF is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_tx*num_streams,
            # num_tx*num_streams]

            # Step 4: Compute A
            # Calculate MMSE Filter (efficiently)
            # W^H = A^-1 H^H
            # A = H^H H \Lambda + N_0 I_Mt
            # \Lambda_ii is a diagonal matrix with \Lambda_ii = E_i = error_var
            if self._rv_domain:
                error_var = tf.concat([error_var, error_var], axis=-2)
                _error_var_row_vec = tf.linalg.matrix_transpose(error_var)
                _A = _G_rv * _error_var_row_vec
            else:
                _error_var_row_vec = tf.linalg.matrix_transpose(error_var)
                _A = G * tf.cast(_error_var_row_vec, dtype=self.dtype)


        # compute LMMSE filter (unit power Tx signals, after noise whitening)
        # _A is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, ...
        # num_streams_per_rx]
        # _I_NT is [1, 1, 1, 1, num_streams_per_rx, num_streams_per_rx]
        _I_NT = tf.linalg.eye(tf.shape(_A)[-1], dtype=_A.dtype)
        _I_NT = tf.reshape(_I_NT, tf.concat([[1] * (_A._rank() - 2), tf.shape(_I_NT)], 0))
        # thermal noise is identity after noise whitening
        _A = _A + _I_NT  # complexity: N_T^2 complex MUL

        # Step 5: compute MMSE filter and outputs, calculate A\H^H
        # A_inv is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, ...
        # num_streams_per_rx]
        # calculating inverse explicitly is EVIL but necessary
        A_inv = tf.linalg.inv(_A)

        if self._rv_domain:
            _G_rv_trans = tf.linalg.matrix_transpose(_G_rv)
            mu_i = tf.math.real(tf.reduce_sum(A_inv * _G_rv_trans, axis=-1, keepdims=True))
        else:
            # G and [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, ...
            # num_streams_per_rx]
            _G_trans = tf.linalg.matrix_transpose(G)
            # equalized signal estimate's error variance
            mu_i = tf.math.real(tf.reduce_sum(A_inv * _G_trans, axis=-1, keepdims=True))
            # mu_i is [batch_size, 1, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, 1]

        rho_i = tf.divide(mu_i, tf.maximum(1 - error_var * mu_i, self._epsilon))

        if self._rv_domain:
            y_hat_i_MF = tf.concat([tf.math.real(y_hat_i_MF), tf.math.imag(y_hat_i_MF)], axis=-2)

        if llr_a is not None:
            y_hat_i_MF_trans = tf.linalg.matrix_transpose(y_hat_i_MF)
            if self._rv_domain:
                y_hat_i_MF_trans = tf.concat([y_hat_i_MF_trans, y_hat_i_MF_trans], axis=-2)
            # efficient parallel equalization after PIC
            z_i = tf.squeeze(
                tf.reduce_sum(A_inv * y_hat_i_MF_trans, axis=-1, keepdims=True) / tf.cast(mu_i, dtype=A_inv.dtype),
                axis=-1)

            ### LMMSE equalization done => continue with LLR calculation

            # Step 6: calculate LLRs
            # calculate exponents
            # Compute squared distances from y to all points

            # log_P_C is [batch_size, num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers,
            # num_constellation] transform log_P_C to [batch_size, 1, num_ofdm_symbols, num_effective_subcarriers,
            # num_tx*num_streams, num_constellation]
            if log_P_C is not None:
                log_P_C = tf.transpose(log_P_C, [0, 3, 4, 1, 2, 5])
                log_P_C_int_shape = tf.concat(
                    [[batch_size], [1], [num_ofdm_data_symbols], [num_effective_subcarriers], [num_tx * num_streams],
                     [num_points]], 0)
                log_P_C = tf.reshape(log_P_C, log_P_C_int_shape)

            z_i = tf.expand_dims(z_i, axis=-1)
        else:
            z_i = tf.linalg.matmul(A_inv, y_hat_i_MF) / tf.cast(mu_i, dtype=A_inv.dtype)

        if self._rv_domain:
            z_i = tf.complex(tf.gather(z_i,tf.range(num_tx),axis=-2), tf.gather(z_i,num_tx+tf.range(num_tx),axis=-2))
            # imaginary and real valued filter bias are the same due to symmetry also in real valued domain
            rho_i = tf.gather(rho_i, tf.range(num_tx), axis=-2)

        # z_i is [batch_size, num_rx, num_ofdm_data_symbols, num_effective_subcarriers, num_streams_per_rx, 1]
        llr_d = self.LLRCalculation(z_i, rho_i, points_reshaped, log_P_C)

        # internal llr_a shape [batch_size, num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers,
        # num_bits_per_symbol]
        # outer llr_a shape is [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol]
        # convert llr_d to out-shape
        llr_d = tf.squeeze(llr_d, axis=[1])
        tmp_shape = tf.concat([[batch_size], [num_ofdm_data_symbols], [num_effective_subcarriers], [num_tx],
                               [num_streams], [num_bits_per_symbol]], 0)
        llr_d = tf.reshape(llr_d, tmp_shape)
        llr_d = tf.transpose(llr_d, [0, 3, 4, 1, 2, 5])
        out_shape = tf.concat([[batch_size], [num_tx], [num_streams], [num_data_symbols * num_bits_per_symbol]], 0)
        llr_d = tf.reshape(llr_d, out_shape)

        # subtract llr_a => llr_e = llr_d - llr_a
        if self._low_complexity:
            # neglects prior (i.e. already extrinsic)
            llr_e = llr_d
        else:
            llr_e = llr_d - llr_a_out

        return [llr_e, y_MF, h_dt_desired_whitened, G]


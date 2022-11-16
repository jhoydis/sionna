#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Simple simulation example that implements an iterative detection and decoding MIMO receiver with MMSE PIC [CST2011]_

[CST2011]_ C. Studer, S. Fateh, and D. Seethaler, "ASIC Implementation of Soft-Input Soft-Output
MIMO Detection Using MMSE Parallel Interference Cancellation," IEEE Journal of Solid-State Circuits,
vol. 46, no. 7, pp. 1754–1765, July 2011. https://ieeexplore.ieee.org/document/5779722
"""

# Load the required sionna components
import tensorflow as tf
import sionna
from tensorflow.keras import Model
from sionna.mimo import StreamManagement, lmmse_equalizer
from sionna.mimo.equalization import SiSoMmsePicEqualizer

import numpy as np

import matplotlib.pyplot as plt
from sionna.mapping import Constellation, Mapper, Demapper, DemapperWithPrior
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.utils import BinarySource, ebnodb2no, sim_ber, expand_to_rank, flatten_dims
from sionna.channel import FlatFadingChannel

#####################################################################################################################
## Simulation Parameters
#####################################################################################################################

# set the total number of LDPC iterations to study
num_ldpc_iter = 10
perfect_csi = False
GPU_NUM = 0

# Debug => smaller batchsize, Monte Carlo iterations
DEBUG = False

channel_model_str = "UMi"  # "UMi", "UMa", "RMa", "Rayleigh"
normalizing_channels = True
low_complexity = True
XLA_ENA = True
OPTIMIZED_LDPC_INTERLEAVER = True

# LoS True only line of sight, False: none-los, none: mix of los and none-los
LoS = True
MOBILITY = True
Antenna_Array = "Dual-Pol-ULA"

# Select GPU 0 to run TF/Sionna
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = GPU_NUM  # Number of the GPU to be used
    try:
        # tf.config.set_visible_devices([], 'GPU')
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)

# Set seeds (TF and NP)
tf.random.set_seed(1)
np.random.seed(1)

dtype = tf.dtypes.complex64

# simulation parameters
batch_size = int(1e3)  # number of symbols to be analyzed
num_iter = 10  # number of Monte Carlo Iterations (total number of Monte Carlo runs is num_iter*batch_size)

if low_complexity:
    demapping_method = "maxlog"
    ldpc_cn_update_func = "minsum"
else:
    demapping_method = "app"
    ldpc_cn_update_func = "boxplus"

stepsize = 1
if DEBUG:
    batch_size = int(1e1)
    num_iter = 1
    stepsize = 5
    tf.config.run_functions_eagerly(True)
    sionna.config.xla_compat = False
else:
    tf.config.run_functions_eagerly(False)
    sionna.config.xla_compat = XLA_ENA

num_bits_per_symbol = 4  # bits per modulated symbol, i.e., 2^4 = 16-QAM
_num_const_bits_ldpc = num_bits_per_symbol
if not OPTIMIZED_LDPC_INTERLEAVER:
    _num_const_bits_ldpc = None
num_streams_per_tx = 1
n_ue = 4
n_bs_ant = 4  # 4 BS antennas

num_idd_iter = 3
num_mmse_pic_iter = 1

# LDPC ENCODING DECODING
# LDPC code parameters
r = 0.5  # rate 1/2
n = 1000
k = int(n * r)  # number of information bits per codeword

# Constellation 16 QAM
# initialize mapper (and demapper) for constellation object
constellation = Constellation("qam", num_bits_per_symbol=num_bits_per_symbol, dtype=dtype)

# Define MU-MIMO System
rx_tx_association = np.zeros([1, n_ue])
rx_tx_association[0, :] = 1

# stream management stores a mapping from Rx and Tx
sm = StreamManagement(rx_tx_association, num_streams_per_tx)

#####################################################################################################################
## Define Models
#####################################################################################################################
class BaseModel(Model):
    def __init__(self, num_bp_iter=5, perfect_csi=False, loss_fun="BCE", training=False):
        super().__init__()
        num_bp_iter = int(num_bp_iter)
        ######################################
        ## Transmitter
        self._binary_source = BinarySource(dtype=dtype.real_dtype)
        self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol=_num_const_bits_ldpc, dtype=dtype.real_dtype)
        self._mapper = Mapper(constellation=constellation, dtype=dtype)
        self._lossFun = loss_fun
        self._training = training

        ######################################
        ## Channel
        self._channel = FlatFadingChannel(num_tx_ant=n_ue, num_rx_ant=n_bs_ant, add_awgn=True, return_channel=True, dtype=dtype)
        self._demapper = Demapper(demapping_method=demapping_method, constellation=constellation, dtype=dtype)


    def computeLoss(self, b, c, b_hat):
        if self._training:
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(c, b_hat))
        else:
            return b, b_hat  # Ground truth and reconstructed information bits returned for BER/BLER computation

class LmmseBaselineModel(BaseModel):
    def __init__(self, num_bp_iter=5, perfect_csi=False):
        super().__init__(num_bp_iter, perfect_csi=perfect_csi)
        ######################################
        ## Receiver
        self._equalizer = lmmse_equalizer
        self._LDPCDec0 = LDPC5GDecoder(self._encoder, cn_type=ldpc_cn_update_func, return_infobits=True,
                                       num_iter=int(num_bp_iter), hard_out=True,
                                       output_dtype=dtype.real_dtype)

    @tf.function(jit_compile=XLA_ENA)
    def call(self, batch_size, ebno_db):
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, r)
        # Outer coding is only performed if not training
        b = self._binary_source([batch_size, n_ue, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        # flattening because coded x tensor has 4 dimensions and FlatFadingChannel only takes in 2 dimensional tensor
        x_flatten = flatten_dims(tf.transpose(x, [0, 2, 1]), 2, 0)
        no_ = expand_to_rank(no, tf.rank(x))
        no_flattened = flatten_dims(tf.transpose(no_*tf.ones(tf.concat([[1,1], tf.shape(x)[2:]], axis=0)), [0, 2, 1]), 2, 0)

        y, h = self._channel([x_flatten, no_flattened])

        s = tf.cast(expand_to_rank(no_flattened, 3, -1) * expand_to_rank(tf.eye(n_bs_ant, n_bs_ant), 3, 0), y.dtype)
        ######################################
        ## Receiver
        [x_hat, no_eff] = self._equalizer(y, h, s)
        llr_ch = self._demapper([x_hat, no_eff])
        llr_ch = flatten_dims(tf.transpose(tf.reshape(llr_ch, [batch_size, int(n/num_bits_per_symbol), n_ue, num_bits_per_symbol]), [0, 2, 1, 3]), 2, 2)
        b_hat = self._LDPCDec0(llr_ch)

        return self.computeLoss(b, c, b_hat)  # Ground truth and reconstructed information bits returned for BER/BLER computation


# IDD model with SISO MMSE PIC Detector [CST2011]_
class iddMmsePicModel(BaseModel):
    def __init__(self, training=False, num_bp_iter=5, perfect_csi=False, num_idd_iter=3):
        super().__init__(num_bp_iter, perfect_csi=perfect_csi, training=training)

        assert num_idd_iter > 1 and isinstance(num_idd_iter, int)

        self._num_idd_iter = num_idd_iter

        self._lmmse_equalizer = lmmse_equalizer
        self._mmse_pic_equalizer = SiSoMmsePicEqualizer(num_iter=num_mmse_pic_iter, constellation=constellation, dtype=dtype)
        self._LDPCDec0 = LDPC5GDecoder(self._encoder, cn_type=ldpc_cn_update_func, return_infobits=False,
                                               num_iter=int(num_bp_iter), stateful=False, hard_out=False,
                                       output_dtype=dtype.real_dtype)
        self._LDPCDec1 = LDPC5GDecoder(self._encoder, cn_type=ldpc_cn_update_func, return_infobits=True,
                                               num_iter=int(num_bp_iter), stateful=False, hard_out=True,
                                            output_dtype=dtype.real_dtype)
        self._demapper_prior = DemapperWithPrior(demapping_method=demapping_method, constellation=constellation, dtype=dtype)

    @tf.function(jit_compile=XLA_ENA)
    def call(self, batch_size, ebno_db):
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, r)
        # Outer coding is only performed if not training
        b = self._binary_source([batch_size, n_ue, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        x_flatten = flatten_dims(tf.transpose(x, [0, 2, 1]), 2, 0)
        no_ = expand_to_rank(no, tf.rank(x))
        no_flattened = flatten_dims(tf.transpose(no_*tf.ones(tf.concat([[1,1], tf.shape(x)[2:]], axis=0)), [0, 2, 1]), 2, 0)

        y, h = self._channel([x_flatten, no_flattened])

        s = tf.cast(expand_to_rank(no_flattened, 3, -1) * expand_to_rank(tf.eye(n_bs_ant, n_bs_ant), 3, 0), y.dtype)
        ######################################
        ## Receiver

        [x_hat, no_eff] = self._lmmse_equalizer(y, h, s)
        llr_ch = self._demapper([x_hat, no_eff])
        llr_ch = flatten_dims(
            tf.transpose(tf.reshape(llr_ch, [batch_size, int(n / num_bits_per_symbol), n_ue, num_bits_per_symbol]),
                         [0, 2, 1, 3]), 2, 2)

        y = tf.reshape(y, [batch_size, int(n/num_bits_per_symbol), n_bs_ant])
        h = tf.reshape(h, [batch_size, int(n/num_bits_per_symbol), n_bs_ant, n_ue])
        s = tf.reshape(s, [batch_size, int(n/num_bits_per_symbol), n_bs_ant, n_bs_ant])

        def idd_iter(llr_ch, it):
            it += 1
            llr_dec = self._LDPCDec0(llr_ch)

            # prior for MMSE PIC are intrinsic LLRs from decoder
            prior = tf.transpose(tf.reshape(llr_dec, [batch_size, n_ue, int(n/num_bits_per_symbol), num_bits_per_symbol]), [0,2,1,3])
            [x_hat, no_eff] = self._mmse_pic_equalizer([y, h, prior, s])

            # MMSE PIC self-iterations (including the approximate PME, i.e, soft-estimate to LLR and LLR to soft-estimate calculation)
            #prior = tf.reshape(self._demapper_prior([x_hat, prior, no_eff]), [batch_size, int(n/num_bits_per_symbol), n_ue, num_bits_per_symbol])
            #[x_hat, no_eff] = self._mmse_pic_equalizer([y, h, prior, s])

            llr_ch = tf.reshape(self._demapper_prior([x_hat, prior, no_eff]), [batch_size, int(n/num_bits_per_symbol), n_ue, num_bits_per_symbol]) - prior    # extrinsic LLRs
            llr_ch = flatten_dims(tf.transpose(llr_ch, [0,2,1,3]), 2, 2)

            return llr_ch, it

        def idd_stop(llr_ch, it):
            return tf.less(it, self._num_idd_iter - 1)

        it = tf.constant(0)
        llr_ch, _ = tf.while_loop(idd_stop, idd_iter, (llr_ch, it), parallel_iterations=1,
                                  maximum_iterations=self._num_idd_iter - 1)

        b_hat = self._LDPCDec1(llr_ch)
        return self.computeLoss(b, c, b_hat)


#####################################################################################################################
## Define Benchmark Models
#####################################################################################################################
# LMMSE Baseline Model
lmmse_baseline = LmmseBaselineModel(num_bp_iter=num_ldpc_iter, perfect_csi=perfect_csi)

# I=3 IDD MMSE PIC Model
idd_mmse_pic_model = iddMmsePicModel(num_bp_iter=int(num_ldpc_iter), num_idd_iter=num_idd_iter, perfect_csi=perfect_csi)

#####################################################################################################################
## Benchmark Models
#####################################################################################################################
snr_range=np.arange(-10, 10+stepsize, stepsize)

BLER = {'snr_range': snr_range}
BER = {'snr_range': snr_range}

title = "Benchmark w. Perfect-CSI=" + str(perfect_csi) + " " + str(n_bs_ant) + 'x' + str(n_ue) + channel_model_str + ' w. 16QAM Mapping & ' + str(num_ldpc_iter) + ' LDPC Iter ' + Antenna_Array

models = [idd_mmse_pic_model, lmmse_baseline]
model_names = ["idd_mmse_pic_model", "lmmse_baseline"]

for i in range(len(models)):
    ber, bler = sim_ber(models[i], ebno_dbs=snr_range, batch_size=batch_size,
                        num_target_block_errors=None, max_mc_iter=num_iter, early_stop=False)
    BLER[model_names[i]] = bler.numpy()
    BER[model_names[i]] = ber.numpy()

# Plot results

plt.figure(figsize=(10, 6))
for i in range(len(models)):
    plt.semilogy(BLER["snr_range"], BLER[model_names[i]], 'o-', c=f'C'+str(i), label=model_names[i])
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()
plt.title(title+" BLER")
plt.show()

plt.figure(figsize=(10, 6))
for i in range(len(models)):
    plt.semilogy(BER["snr_range"], BER[model_names[i]], 'o-', c=f'C'+str(i), label=model_names[i])
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()
plt.title(title+" BLER")
plt.show()

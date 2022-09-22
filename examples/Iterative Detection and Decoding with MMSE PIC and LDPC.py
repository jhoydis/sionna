#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition and functions related to OFDM channel equalization"""

# Load the required sionna components
import tensorflow as tf
import sionna
from sionna.channel.tr38901 import PanelArray, UMi, UMa, RMa
from tensorflow.keras import Model
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LMMSEEqualizer, LSChannelEstimator, PilotPattern

import numpy as np

import matplotlib.pyplot as plt
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.ofdm.equalization import SisoMmsePicDetector
from sionna.utils import BinarySource, ebnodb2no, sim_ber, expand_to_rank
from sionna.channel import RayleighBlockFading, OFDMChannel, gen_single_sector_topology

#####################################################################################################################
## Simulation Parameters
#####################################################################################################################

# set the total number of LDPC iterations to study
num_ldpc_iter = 10
perfect_csi = False
GPU_NUM = 0

# Debug => smaller batchsize, Monte Carlo iterations
DEBUG = True

channel_model_str = "UMi"  # "UMi", "UMa", "RMa", "Rayleigh"
normalizing_channels = True
low_complexity = True
XLA_ENA = False
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

# OFDM Waveform Settings
# 30 kHz subcarrier spacing
subcarrier_spacing = 30e3
# Maximum resource blocks within the 100MHz n78 5G channel w/ 30MHz sc spacing
num_resource_blocks = 273
num_subcarriers = num_resource_blocks * 12
# we use only a subset of resource blocks for our simulations (here, some smaller number 2-15)
rb_used = 5
sc_used = rb_used * 12

carrier_freq = 3.75e9   # 3.7GHz-3.8GHz n78 100MHz band
bandwidth = 100e6
# effective_bandwidth = subcarrier_spacing * num_subcarriers
# 14 OFDM TIME symbols is one 5G OFDM frame
num_ofdm_symbols = 14
num_pilot_symbols = 4

# simulation parameters
batch_size = int(1e1)  # number of symbols to be analyzed
num_iter = 10  # number of Monte Carlo Iterations (total number of Monte Carlo runs is num_iter*batch_size)

if low_complexity:
    demapping_method = "maxlog"
    ldpc_cn_update_func = "minsum"
else:
    demapping_method = "app"
    ldpc_cn_update_func = "boxplus"

stepsize = 2.5
if DEBUG:
    batch_size = int(1e1)
    num_iter = 1
    stepsize = 5
    tf.config.run_functions_eagerly(True)
    sionna.config.xla_compat = True
    rb_used = 1
else:
    tf.config.run_functions_eagerly(False)
    sionna.config.xla_compat = XLA_ENA

num_bits_per_symbol = 4  # bits per modulated symbol, i.e., 2^4 = 16-QAM
_num_const_bits_ldpc = num_bits_per_symbol
if not OPTIMIZED_LDPC_INTERLEAVER:
    _num_const_bits_ldpc = None
num_streams_per_tx = 1
n_ue = 4
n_bs_ant = 16  # 4 BS antennas

num_idd_iter = 3


if channel_model_str in ["UMi", "UMa", "RMa"]:
    bs_array = None
    if Antenna_Array == "4x4":
        bs_array = PanelArray(num_rows_per_panel=int(n_bs_ant / 4),
                              num_cols_per_panel=4,
                              polarization='single',
                              polarization_type='V',
                              antenna_pattern='38.901',
                              carrier_frequency=carrier_freq)
    elif Antenna_Array == "Single-Pol-ULA":
        bs_array = PanelArray(num_rows_per_panel=1,
                              num_cols_per_panel=n_bs_ant,
                              polarization='single',
                              polarization_type='V',
                              antenna_pattern='38.901',
                              carrier_frequency=carrier_freq)
    elif Antenna_Array == "Single-Pol-Omni-ULA":
        bs_array = PanelArray(num_rows_per_panel=1,
                              num_cols_per_panel=n_bs_ant,
                              polarization='single',
                              polarization_type='V',
                              antenna_pattern='omni',
                              carrier_frequency=carrier_freq)
    elif Antenna_Array == "Dual-Pol-ULA":
        bs_array = PanelArray(num_rows_per_panel=1,
                              num_cols_per_panel=int(n_bs_ant/2),
                              polarization='dual',
                              polarization_type='cross',
                              antenna_pattern='38.901',
                              carrier_frequency=carrier_freq)
    else:
        bs_array = PanelArray(num_rows_per_panel=int(n_bs_ant/2/8),
            num_cols_per_panel = 8,
            polarization = 'dual',
            polarization_type = 'cross',
            antenna_pattern = '38.901',
            carrier_frequency = carrier_freq)
    ut_array = PanelArray(num_rows_per_panel=1,
        num_cols_per_panel = 1,
        polarization = 'single',
        polarization_type = 'V',
        antenna_pattern = 'omni',
        carrier_frequency = carrier_freq)
elif channel_model_str =="Rayleigh":
    # channel_model = RayleighBlockFading(num_rx=1, num_rx_ant=n_bs_ant, num_tx=n_ue, num_tx_ant=1)
    # channel_model_str = " Frequency-Flat Rayleigh-Block-Fading "
    pass
else:
    raise NameError('channel_model_string not found')

max_ut_velocity = 50.0/3.6
if not MOBILITY:
    max_ut_velocity = 0

# LDPC ENCODING DECODING
# LDPC code parameters
r = 0.5  # rate 1/2
n = int(
    sc_used * (num_ofdm_symbols - num_pilot_symbols) * num_bits_per_symbol)  # code length (most probably selects the largest (2nd) 5G generator matrix
k = int(n * r)  # number of information bits per codeword

# Constellation 16 QAM
# initialize mapper (and demapper) for constellation object
constellation = Constellation("qam", num_bits_per_symbol=num_bits_per_symbol)

# Define MU-MIMO System
rx_tx_association = np.zeros([1, n_ue])
rx_tx_association[0, :] = 1

# stream management stores a mapping from Rx and Tx
sm = StreamManagement(rx_tx_association, num_streams_per_tx)

# pilot pattern that samples all subcarriers for each UE
mask = np.zeros([   n_ue, 1, num_ofdm_symbols, sc_used], bool)
mask[...,[2,3,11,12],:] = True
pilots = np.zeros([n_ue, 1, np.sum(mask[0,0])])
pilots[0,0, 0*sc_used:1*sc_used:2] = 1
pilots[0,0, 2*sc_used:3*sc_used:2] = 1
pilots[1,0,1+0*sc_used:1*sc_used:2] = 1
pilots[1,0,1+2*sc_used:3*sc_used:2] = 1
pilots[2,0, 1*sc_used:2*sc_used:2] = 1
pilots[2,0, 3*sc_used:4*sc_used:2] = 1
pilots[3,0,1+1*sc_used:2*sc_used:2] = 1
pilots[3,0,1+3*sc_used:4*sc_used:2] = 1
pilot_pattern = PilotPattern(mask, pilots, normalize=True)

if DEBUG:
    with tf.device("/cpu:0"):  # weird tensorflow-macos (Apple M1) for missing Apple GPU implementations
        rg_chan_est = ResourceGrid(num_ofdm_symbols=14, fft_size=sc_used,
                          subcarrier_spacing=subcarrier_spacing, cyclic_prefix_length=20,
                          num_tx=n_ue, pilot_ofdm_symbol_indices=[2,11],
                          num_streams_per_tx=num_streams_per_tx, pilot_pattern=pilot_pattern,
                          )
        rg_chan_est.show()
        rg_chan_est.pilot_pattern.show()
        plt.show()
else:
    rg_chan_est = ResourceGrid(num_ofdm_symbols=14, fft_size=sc_used,
                               subcarrier_spacing=subcarrier_spacing, cyclic_prefix_length=20,
                               num_tx=n_ue, pilot_ofdm_symbol_indices=[2, 11],
                               num_streams_per_tx=num_streams_per_tx, pilot_pattern=pilot_pattern,
                               )
rg_chan_est.show()
rg_chan_est.pilot_pattern.show()
plt.show()
#####################################################################################################################
## Define Models
#####################################################################################################################
class BaseModel(Model):
    def __init__(self, num_bp_iter=5, perfect_csi=False, loss_fun="BCE", training=False):
        super().__init__()
        num_bp_iter = int(num_bp_iter)
        ######################################
        ## Transmitter
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol=_num_const_bits_ldpc)
        self._mapper = Mapper(constellation=constellation)
        self._rg_mapper = ResourceGridMapper(rg_chan_est)
        self._lossFun = loss_fun
        self._training = training

        ######################################
        ## Receiver
        self._ls_est = LSChannelEstimator(rg_chan_est, interpolation_type="lin")
        self._perfect_csi = perfect_csi

        ######################################
        ## Channel
        if channel_model_str == "UMi":
            self._channel_model = UMi(carrier_frequency=carrier_freq,
                                o2i_model='low',
                                ut_array=ut_array,
                                bs_array=bs_array,
                                direction='uplink')
        elif channel_model_str == "UMa":
            self._channel_model = UMa(carrier_frequency=carrier_freq,
                                      o2i_model='low',
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction='uplink')
        elif channel_model_str == "RMa":
            self._channel_model = RMa(carrier_frequency=carrier_freq,
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction='uplink')
        elif channel_model_str == "Rayleigh":
            self._channel_model = RayleighBlockFading(num_rx=1, num_rx_ant=n_bs_ant, num_tx=n_ue, num_tx_ant=1)
        self._channel = OFDMChannel(channel_model=self._channel_model,
                                    resource_grid=rg_chan_est,
                                    add_awgn=True,
                                    normalize_channel=normalizing_channels, return_channel=True)
    def new_topology(self, batch_size):
        """Set new topology"""
        if channel_model_str in ["UMi", "UMa", "RMa"]:
            # sensible values according to 3GPP standard, no mobility by default
            topology = gen_single_sector_topology(batch_size,
                                                  n_ue, max_ut_velocity=max_ut_velocity,
                                                  scenario=channel_model_str.lower())
            self._channel_model.set_topology(*topology, los=LoS)

    def computeLoss(self, b, c, b_hat):
        if self._training:
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(c, b_hat))
        else:
            return b, b_hat  # Ground truth and reconstructed information bits returned for BER/BLER computation

class LmmseBaselineModelChanEst(BaseModel):
    def __init__(self, num_bp_iter=5, perfect_csi=False):
        super().__init__(num_bp_iter, perfect_csi=perfect_csi)
        ######################################
        ## Receiver
        self._equalizer = LMMSEEqualizer(rg_chan_est, sm)
        self._demapper = Demapper(demapping_method=demapping_method, constellation=constellation)
        self._LDPCDec0 = LDPC5GDecoder(self._encoder, cn_type=ldpc_cn_update_func, return_infobits=True,
                                       num_iter=int(num_bp_iter), hard_out=True)

    @tf.function(jit_compile=XLA_ENA)
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, r)
        # Outer coding is only performed if not training
        b = self._binary_source([batch_size, n_ue, num_streams_per_tx, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel([x_rg, no_])

        ######################################
        ## Receiver
        if self._perfect_csi:
            h_hat = h
            chan_est_var = tf.zeros(tf.shape(h_hat), dtype=tf.float32)  # No channel estimation error when perfect CSI knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est([y, no])

        [x_hat, no_eff] = self._equalizer([y, h_hat, chan_est_var, no])
        llr_ch = self._demapper([x_hat, no_eff])
        b_hat = self._LDPCDec0(llr_ch)

        return self.computeLoss(b, c, b_hat)  # Ground truth and reconstructed information bits returned for BER/BLER computation


# IDD model with baseline siso mmse pic detector
class iddMmsePicChanEst(BaseModel):
    def __init__(self, training=False, num_bp_iter=5, perfect_csi=False, num_idd_iter=3):
        super().__init__(num_bp_iter, perfect_csi=perfect_csi, training=training)

        assert num_idd_iter > 1 and isinstance(num_idd_iter, int)

        self._num_idd_iter = num_idd_iter

        self._detector0 = SisoMmsePicDetector(rg_chan_est, sm, demapping_method=demapping_method,
                                                 constellation=constellation, low_complexity=low_complexity,
                                                  data_carrying_whitened_inputs=False)
        self._detector1 = SisoMmsePicDetector(rg_chan_est, sm, demapping_method=demapping_method,
                                                 constellation=constellation, low_complexity=low_complexity,
                                                  data_carrying_whitened_inputs=True)
        self._LDPCDec0 = LDPC5GDecoder(self._encoder, cn_type=ldpc_cn_update_func, return_infobits=False,
                                               num_iter=int(num_bp_iter), stateful=False, hard_out=False)
        self._LDPCDec1 = LDPC5GDecoder(self._encoder, cn_type=ldpc_cn_update_func, return_infobits=True,
                                               num_iter=int(num_bp_iter), stateful=False, hard_out=True)

    @tf.function(jit_compile=XLA_ENA)
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, r)
        # Outer coding is only performed if not training
        b = self._binary_source([batch_size, n_ue, num_streams_per_tx, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel([x_rg, no_])

        ######################################
        ## Receiver
        if self._perfect_csi:
            h_hat = h
            chan_est_var = tf.zeros(tf.shape(h_hat), dtype=tf.float32)  # No channel estimation error when perfect CSI knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est([y, no])

        [llr_ch, y_MF, h_dt_desired_whitened, G] = self._detector0(
            [y, h_hat, chan_est_var, no, None, None])

        def idd_iter(llr_ch, it):
            it += 1
            llr_dec = self._LDPCDec0(llr_ch)
            [llr_ch, _, _, _] = self._detector1([y_MF, h_dt_desired_whitened, chan_est_var, no, llr_dec, G])
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
lmmse_baseline = LmmseBaselineModelChanEst(num_bp_iter=num_ldpc_iter, perfect_csi=perfect_csi)

# I=3 IDD MMSE PIC Model
three_idd_iter_mmse_pic_model = iddMmsePicChanEst(num_bp_iter=int(num_ldpc_iter), num_idd_iter=num_idd_iter, perfect_csi=perfect_csi)

#####################################################################################################################
## Benchmark Models
#####################################################################################################################
snr_range=np.arange(-10, 10+stepsize, stepsize)

BLER = {'snr_range': snr_range}
BER = {'snr_range': snr_range}

title = "Benchmark w. Perfect-CSI=" + str(perfect_csi) + " " + str(n_bs_ant) + 'x' + str(n_ue) + channel_model_str + ' w. 16QAM Mapping & ' + str(num_ldpc_iter) + ' LDPC Iter ' + Antenna_Array

models = [three_idd_iter_mmse_pic_model, lmmse_baseline]
model_names = ["three_idd_iter_mmse_pic_model", "lmmse_baseline"]

for i in range(len(models)):
    ber, bler = sim_ber(models[i], ebno_dbs=snr_range, batch_size=batch_size,
                        num_target_block_errors=None, max_mc_iter=num_iter, early_stop=False)
    BLER[model_names[i]] = bler.numpy()
    BER[model_names[i]] = ber.numpy()


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

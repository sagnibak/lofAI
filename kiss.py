from conv import GConv
from data_utils import LofiSequence
from layers import ChannelShuffle, ConvBNLReLU, DenseBlock, DenseUpsampling, Sampling, WeightSaver

from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Add, BatchNormalization, Conv2D, Conv2DTranspose, Input, SeparableConv2D, \
                         MaxPool2D, AvgPool2D, Concatenate, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import mse
from keras.models import Model
from keras.regularizers import l2
from keras.utils import plot_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from time import time


spectral_in = Input(shape=(128, 1_024, 2), name='spectral_in')  # 16.384 second long segment


def build_encoder():

    encoded = ConvBNLReLU(16, (4, 8), (2, 4), conv_func=Conv2D)(spectral_in)  # 64 x 256 x 16

    encoded0 = ConvBNLReLU(64, (4, 8), (2, 4), conv_func=Conv2D)(encoded)  # 32 x 64 x 64
    encoded = ConvBNLReLU(80, (4, 8), (1, 1))(encoded0)  # 32 x 64 x 80

    encoded = ConvBNLReLU(96, (3, 6), (2, 4))(encoded)  # 16 x 16 x 96

    encoded0 = AvgPool2D(pool_size=(2, 4), padding='same')(encoded0)
    encoded = Concatenate()([encoded, encoded0])  # 16 x 16 x 160

    encoded1 = ConvBNLReLU(128, (3, 3), (2, 2))(encoded)  # 8 x 8 x 128
    encoded = ConvBNLReLU(160, (3, 3), (2, 2))(encoded1)  # 4 x 4 x 160

    encoded0 = AvgPool2D(pool_size=(4, 4), padding='same')(encoded0)
    encoded = Concatenate()([encoded, encoded0])  # 4 x 4 x 224

    encoded = ConvBNLReLU(192, (2, 2), (2, 2))(encoded)  # 2 x 2 x 192

    encoded1 = AvgPool2D(pool_size=(4, 4), padding='same')(encoded1)
    encoded = Concatenate()([encoded, encoded1])  # 2 x 2 x 320

    encoded = ConvBNLReLU(256, (2, 2), (2, 2))(encoded)  # 1 x 1 x 256

    z_mean = GConv(1, 256, 16, kernel_regularizer=l2())(encoded)
    z_log_var = GConv(1, 256, 16, kernel_regularizer=l2())(encoded)
    z_sampled = Sampling('z_sampled')([z_mean, z_log_var])

    return Model(spectral_in, [z_mean, z_log_var, z_sampled], name='encoder'), z_mean, z_log_var


def build_decoder():
    z_sampled_in = Input(shape=(1, 1, 16), name='z_sampled_in')

    decoded0 = ConvBNLReLU(256, (1, 1), (1, 1), conv_func=Conv2D)(z_sampled_in)  # 1 x 1 x 256

    decoded0 = UpSampling2D(size=(2, 2))(decoded0)  # 2 x 2 x 256
    decoded = ConvBNLReLU(192, (2, 2), (1, 1))(decoded0)  # 2 x 2 x 192

    decoded = UpSampling2D(size=(2, 2))(decoded)  # 4 x 4 x 192
    decoded1 = ConvBNLReLU(160, (2, 2), (1, 1))(decoded)  # 4 x 4 x 160

    decoded0 = UpSampling2D(size=(4, 4))(decoded0)  # 8 x 8 x 192
    decoded = UpSampling2D(size=(2, 2))(decoded1)  # 8 x 8 x 160

    decoded = Concatenate()([decoded, decoded0])  # 8 x 8 x 352
    decoded = ConvBNLReLU(128, (3, 3), (1, 1))(decoded)  # 8 x 8 x 128

    decoded1 = UpSampling2D(size=(4, 4))(decoded1)  # 16 x 16 x 192
    decoded = UpSampling2D(size=(2, 2))(decoded)  # 16 x 16 x 128

    decoded = Concatenate()([decoded1, decoded])
    decoded = ConvBNLReLU(80, (3, 3), (1, 1))(decoded)  # 16 x 16 x 80

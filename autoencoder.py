from layers import ConvBNLReLU, DenseBlock

from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Input, SeparableConv2D, \
                         MaxPool2D, AvgPool2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.regularizers import l2
from keras.utils import plot_model

spectral_in = Input(shape=(128, 1_024, 2), name='spectral_in')  # 16.384 second long segment

# encoded = Conv2D(32, (3, 6), strides=(2, 4), kernel_regularizer=l2(),
#                  padding='same', use_bias=False)(spectral_in)
# encoded = LeakyReLU(0.1)(BatchNormalization()(encoded))  # 64 x 256 x 32
#
# encoded = SeparableConv2D(128, (3, 6), strides=(2, 4), kernel_regularizer=l2(),
#                           padding='same', use_bias=False)(encoded)
# encoded_ = LeakyReLU(0.1)(BatchNormalization()(encoded))  # 32 x 64 x 128
#
# encoded = SeparableConv2D(384, (3, 6), strides=(2, 4), kernel_regularizer=l2(),
#                           padding='same', use_bias=False)(encoded_)
# encoded = LeakyReLU(0.1)(BatchNormalization()(encoded))  # 16 x 16 x 384
#
# encoded_ = AvgPool2D((4, 8), strides=(4, 8), padding='same')(encoded_)  # 8 x 8 x 128
#
# encoded = SeparableConv2D(512, (3, 3), strides=(2, 2), kernel_regularizer=l2(),
#                           padding='same', use_bias=False)(encoded)
# encoded = LeakyReLU(0.1)(BatchNormalization()(encoded))  # 8 x 8 x 512
#
# encoded = SeparableConv2D(640, (3, 3), strides=(2, 2), kernel_regularizer=l2(),
#                           padding='same', use_bias=False)(Concatenate()([encoded, encoded_]))
#


encoded = ConvBNLReLU(32, (3, 6), strides=(2, 4), conv_func=Conv2D)(spectral_in)  # 64 x 256 x 32
encoded = ConvBNLReLU(128, (3, 6), strides=(2, 4), conv_func=Conv2D)(encoded)  # 32 x 64 x 128
encoded = DenseBlock(128, (3, 3), num_layers=3,
                     first_kernel=(3, 6), first_stride=(2, 4),
                     pool_func_args={'pool_size': (3, 6), 'strides': (2, 4)})(encoded)  # 16 x 16 x 256
encoded = DenseBlock(256, (3, 3), num_layers=3, first_stride=(2, 2),
                     pool_func_args={'pool_size': (3, 3), 'strides': (2, 2)})(encoded)  # 8 x 8 x 512
encoded = DenseBlock(256, (3, 3), num_layers=3, first_stride=(2, 2),
                     pool_func_args={'pool_size': (3, 3), 'strides': (2, 2)})(encoded)  # 4 x 4 x 768
encoded = DenseBlock(256, (3, 3), num_layers=3, first_stride=(2, 2),
                     pool_func_args={'pool_size': (2, 2), 'strides': (2, 2)})(encoded)  # 2 x 2 x 1024

encoder = Model(spectral_in, encoded, name='encoder')
encoder.summary()
plot_model(encoder, 'proto_enc.png', show_shapes=True)

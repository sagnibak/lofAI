from conv import GConv
from data_utils import LofiSequence
from layers import ChannelShuffle, ConvBNLReLU, DenseBlock, DenseUpsampling, Sampling, WeightSaver

from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Add, BatchNormalization, Conv2D, Conv2DTranspose, Input, SeparableConv2D, \
                         MaxPool2D, AvgPool2D, Concatenate
from keras.losses import mse
from keras.models import Model
from keras.regularizers import l2
from keras.utils import plot_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from time import time

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


spectral_in = Input(shape=(128, 1_024, 2), name='spectral_in')  # 16.384 second long segment


def build_encoder():

    encoded = ConvBNLReLU(32, (3, 6), strides=(2, 4), conv_func=Conv2D)(spectral_in)  # 64 x 256 x 32
    encoded = DenseBlock(32, (2, 4), num_layers=2, merge_func=Add)(encoded)

    encoded = ConvBNLReLU(128, (3, 6), strides=(2, 4), conv_func=Conv2D)(encoded)  # 32 x 64 x 128
    encoded = DenseBlock(128, (3, 3), num_layers=3,
                         first_kernel=(3, 6), first_stride=(2, 4),
                         pool_func_args={'pool_size': (3, 6), 'strides': (2, 4)})(encoded)  # 16 x 16 x 256
    encoded = DenseBlock(256, (3, 3), num_layers=2, merge_func=Add)(encoded)

    encoded = DenseBlock(256, (3, 3), num_layers=3, first_stride=(2, 2),
                         pool_func_args={'pool_size': (3, 3), 'strides': (2, 2)})(encoded)  # 8 x 8 x 512

    encoded = DenseBlock(256, (3, 3), num_layers=3, first_stride=(2, 2),
                         pool_func_args={'pool_size': (3, 3), 'strides': (2, 2)})(encoded)  # 4 x 4 x 768
    encoded = DenseBlock(768, (2, 2), num_layers=3, merge_func=Add)(encoded)

    encoded = DenseBlock(256, (3, 3), num_layers=3, first_stride=(2, 2),
                         pool_func_args={'pool_size': (2, 2), 'strides': (2, 2)})(encoded)  # 2 x 2 x 1024
    encoded = DenseBlock(128, (1, 1), num_layers=2)(encoded)  # 2 x 2 x 1152
    encoded = ChannelShuffle(9)(encoded)
    encoded = DenseBlock(128, (1, 1), num_layers=2)(encoded)  # 2 x 2 x 1280
    encoded = ChannelShuffle(10)(encoded)

    z_mean = GConv(4, 1280, 1280, kernel_regularizer=l2())(encoded)
    z_log_var = GConv(4, 1280, 1280, kernel_regularizer=l2())(encoded)
    z_sampled = Sampling('z_sampled')([z_mean, z_log_var])

    return Model(spectral_in, [z_mean, z_log_var, z_sampled], name='encoder'), z_mean, z_log_var


def build_decoder():
    z_sampled_in = Input(shape=(2, 2, 1280), name='z_sampled_in')

    decoded = GConv(2, 1280, 1152, kernel_regularizer=l2())(z_sampled_in)
    decoded = ChannelShuffle(3)(decoded)
    decoded = GConv(2, 1152, 1024, kernel_regularizer=l2())(decoded)  # 2 x 2 x 1024

    decoded = DenseUpsampling(out_channels=768, filters=512, kernel_size=(2, 2), num_layers=3,
                              first_kernel=(3, 3), first_stride=(2, 2),
                              pool_func_args={'size': (2, 2)})(decoded)  # 4 x 4 x 768

    decoded = DenseUpsampling(out_channels=768, filters=384, kernel_size=(2, 2),
                              num_layers=3)(decoded)  # 4 x 4 x 768

    decoded = DenseUpsampling(out_channels=512, filters=384, kernel_size=(2, 2), num_layers=3,
                              first_kernel=(3, 3), first_stride=(2, 2),
                              pool_func_args={'size': (2, 2)})(decoded)  # 8 x 8 x 512

    decoded = DenseUpsampling(out_channels=512, filters=256, kernel_size=(3, 3),
                              num_layers=3)(decoded)  # 8 x 8 x 512

    decoded = DenseUpsampling(out_channels=256, filters=256, kernel_size=(3, 3), num_layers=3,
                              first_kernel=(3, 3), first_stride=(2, 2),
                              pool_func_args={'size': (2, 2)})(decoded)  # 16 x 16 x 256

    decoded = DenseUpsampling(out_channels=256, filters=128, kernel_size=(3, 3),
                              num_layers=2)(decoded)  # 16 x 16 x 256

    decoded = DenseUpsampling(out_channels=128, filters=128, kernel_size=(3, 3),
                              first_kernel=(3, 6), first_stride=(2, 4),
                              pool_func_args={'size': (2, 4)})(decoded)  # 32 x 64 x 128

    decoded = DenseUpsampling(out_channels=64, filters=40, kernel_size=(3, 3),
                              first_kernel=(3, 6), first_stride=(2, 4),
                              pool_func_args={'size': (2, 4)})(decoded)  # 64 x 256 x 64

    decoded = DenseUpsampling(out_channels=8, filters=16, kernel_size=(3, 6),
                              first_kernel=(3, 6), first_stride=(2, 4),
                              pool_func_args={'size': (2, 4)})(decoded)  # 128 x 1024 x 8

    decoded = DenseUpsampling(out_channels=8, filters=8, kernel_size=(3, 6),
                              num_layers=3)(decoded)  # 128 x 1024 x 2

    decoded = Conv2D(2, (3, 3), padding='same', kernel_regularizer=l2())(decoded)

    return Model(z_sampled_in, decoded, name='decoder')


if __name__ == '__main__':

    num_epochs = 1000
    batch_size = 5

    encoder, z_mean, z_log_var = build_encoder()
    encoder_saver = WeightSaver(encoder, 'models/encoders/enc1')

    encoder.summary()
    plot_model(encoder, 'model_plots/proto_enc4.png', show_shapes=True)

    decoder = build_decoder()
    decoder_saver = WeightSaver(decoder, 'models/decoders/dec1')

    print('========================\n'*3)
    decoder.summary()
    plot_model(decoder, 'model_plots/proto_dec2.png', show_shapes=True)

    outs = decoder(encoder(spectral_in)[2])
    vae = Model(spectral_in, outs, name='sound_vae')

    def vae_loss(y_true, y_pred):
        with K.name_scope('Loss'):
            reconstruction_loss = K.sum(mse(K.flatten(y_true), K.flatten(y_pred)) * 128 * 1024)
            kl_loss = 0.5 * (K.sum(K.exp(z_log_var) + K.square(z_mean) - 1 - z_log_var))

            return 0.05*kl_loss + 10*reconstruction_loss

    vae.compile(optimizer='adam', loss=vae_loss)

    tensorboard = TensorBoard(log_dir=f'logs/{time()}')

    lofi_gen = LofiSequence(batch_size=batch_size)
    lofi_val_gen = LofiSequence(batch_size=batch_size, validation=True)

    vae.fit_generator(lofi_gen, epochs=num_epochs,
                      callbacks=[
                          tensorboard,
                          encoder_saver,
                          decoder_saver
                      ],
                      validation_data=lofi_val_gen,
                      workers=6,
                      use_multiprocessing=True,
                      shuffle=True)

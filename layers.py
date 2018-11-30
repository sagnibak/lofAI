from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Add, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, \
                         Lambda, SeparableConv2D, AvgPool2D, MaxPool2D, \
                         UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.models import Model

import numpy as np

from typing import Union


class ConvBNLReLU:
    count = 0

    def __init__(self, filters, kernel_size, strides, alpha=0.1,
                 conv_func: Union[Conv2D, SeparableConv2D]=SeparableConv2D):
        self.filters = filters
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.conv_func = conv_func
        ConvBNLReLU.count += 1

        if strides[0] != strides[1]:
            self.strides = (min(strides), min(strides))
            self.pool_size = (1, int(strides[1] / strides[0]))
            self.pool = True
        else:
            self.strides = strides
            self.pool = False

    def __call__(self, tensor):
        with K.name_scope(f'Conv_BN_LReLU_{ConvBNLReLU.count}'):
            x = self.conv_func(self.filters, self.kernel_size,
                               strides=self.strides, padding='same',
                               kernel_regularizer=l2(),
                               use_bias=False)(tensor)

            if self.pool:
                x = MaxPool2D(pool_size=self.pool_size, padding='same')(x)

            x = BatchNormalization()(x)

            return LeakyReLU(self.alpha)(x)


class DenseBlock:
    count = 0

    def __init__(self, filters, kernel_size,
                 num_layers=3, first_kernel=None, first_stride=(1, 1),
                 pool_func: Union[AvgPool2D, MaxPool2D]=AvgPool2D,
                 pool_func_args: dict=None,
                 merge_func: Union[Add, Concatenate]=Concatenate):
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.first_kernel = kernel_size if first_kernel is None else first_kernel
        self.first_stride = first_stride
        self.pool_func = pool_func
        self.pool_func_args = pool_func_args if pool_func_args is not None else dict()
        self.pool_func_args['padding'] = 'same'
        self.merge_func = merge_func
        DenseBlock.count += 1

    def __call__(self, tensor):
        with K.name_scope(f'Dense_Block_{DenseBlock.count}'):
            x = ConvBNLReLU(self.filters, self.first_kernel, self.first_stride)(tensor)

            # pool only if downsampling
            if self.first_stride != (1, 1):
                x_ = self.pool_func(**self.pool_func_args)(tensor)
            else:
                x_ = tensor

            for i in range(self.num_layers - 1):
                x = ConvBNLReLU(self.filters, self.kernel_size, (1, 1))(x)
            return self.merge_func()([x_, x])


class DenseUpsampling(DenseBlock):
    """This is a `DenseBlock` that performs upsampling instead of upsampling.
    So `first_kernel` and `first_stride` are passed to Conv2DTranspose, and
    `pool_func` needs to be an Upsampling layer.
    """
    count = 0

    def __init__(self, out_channels=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_func = UpSampling2D
        self.pool_func_args.pop('padding')
        self.out_channels = out_channels if out_channels is not None else self.filters

    def __call__(self, tensor):
        with K.name_scope(f'Upsampling_Block_{DenseUpsampling.count}'):
            # x = ConvBNLReLU(self.filters, self.first_kernel, self.first_stride,
            #                 conv_func=Conv2DTranspose)(tensor)
            if self.first_stride != (1, 1):
                x = UpSampling2D(size=self.first_stride)(tensor)
                x_ = self.pool_func(**self.pool_func_args)(tensor)
            else:
                x = tensor
                x_ = tensor

            for i in range(self.num_layers - 1):
                x = ConvBNLReLU(self.filters, self.kernel_size, (1, 1))(x)
            x = self.merge_func()([x_, x])

            return ConvBNLReLU(self.out_channels, self.kernel_size, (1, 1))(x)


class Sampling:

    def __init__(self, name):
        self.name = name

    def __call__(self, args):
        with K.name_scope('Sampling'):
            return Lambda(self._samping_op, name=self.name)(args)

    def _samping_op(self, args):
        z_mean, z_log_var = args
        batch_size = K.shape(z_mean)[0]
        other_dims = K.int_shape(z_mean)[1:]
        eps = K.random_normal(shape=(batch_size, *other_dims),
                              mean=0., stddev=0.1)
        return z_mean + K.exp(0.5 * z_log_var) * eps


class ChannelShuffle:
    count = 0

    def __init__(self, num_groups):
        self.num_groups = num_groups
        ChannelShuffle.count += 1

    def __call__(self, tensor):
        with K.name_scope(f'Channel_Shuffle_{ChannelShuffle.count}'):
            return Lambda(self._shuffle_op)(tensor)

    def _shuffle_op(self, tensor):
        _, h, w, c = K.int_shape(tensor)
        num_ch_per_group = c // self.num_groups

        if num_ch_per_group * self.num_groups != c:
            raise ValueError(f'Number of groups ({self.num_groups}) does not evenly divide '
                             f'the number of channels ({c}) in the tensor.')

        x = K.reshape(x=tensor, shape=(-1, h, w, num_ch_per_group, self.num_groups))
        x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
        x = K.reshape(x, (-1, h, w, c))

        return x


class WeightSaver(Callback):

    def __init__(self, model: Model, filepath: str):
        super().__init__()
        self.model_ = model
        self.filepath = filepath
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < self.best:
            save_path = self.filepath + f'_{epoch + 1:03d}_{logs["val_loss"]:.3f}.h5'
            self.model_.save_weights(save_path)
            print(f'Saved weights at {save_path}')
            self.best = logs['val_loss']

from keras import backend as K
from keras.layers import Add, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, SeparableConv2D, \
                         AvgPool2D, MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2

from typing import Union


class ConvBNLReLU:
    count = 0

    def __init__(self, filters, kernel_size, strides, alpha=0.1,
                 conv_func: Union[Conv2D, SeparableConv2D, Conv2DTranspose]=SeparableConv2D):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.alpha = alpha
        self.conv_func = conv_func
        ConvBNLReLU.count += 1

    def __call__(self, tensor):
        with K.name_scope(f'Conv_BN_LReLU_{ConvBNLReLU.count}'):
            x = self.conv_func(self.filters, self.kernel_size,
                               strides=self.strides, padding='same',
                               kernel_regularizer=l2(),
                               use_bias=False)(tensor)

            x = BatchNormalization()(x)

            return LeakyReLU(self.alpha)(x)


class DenseBlock:
    count = 0

    def __init__(self, filters, kernel_size,
                 num_layers=3, first_kernel=None, first_stride=(1, 1),
                 pool_func: Union[AvgPool2D, MaxPool2D]=AvgPool2D,
                 pool_func_args: dict=None,
                 merge_func: Union[Add, Concatenate]=Concatenate,
                 upsampling=False):
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.first_kernel = kernel_size if first_kernel is None else first_kernel
        self.first_stride = first_stride
        self.pool_func = pool_func
        self.pool_func_args = pool_func_args if pool_func_args is not None else dict()
        self.pool_func_args['padding'] = 'same'
        self.merge_func = merge_func
        self.conv_func = Conv2DTranspose if upsampling else SeparableConv2D
        DenseBlock.count += 1

    def __call__(self, tensor):
        with K.name_scope(f'Dense_Block_{DenseBlock.count}'):
            x = ConvBNLReLU(self.filters, self.first_kernel, self.first_stride,
                            conv_func=self.conv_func)(tensor)
            x_ = self.pool_func(**self.pool_func_args)(tensor)
            for i in range(self.num_layers - 1):
                x = ConvBNLReLU(self.filters, self.kernel_size, (1, 1),
                                conv_func=self.conv_func)(x)
            return self.merge_func()([x_, x])

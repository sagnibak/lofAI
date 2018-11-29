from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import DepthwiseConv2D, Conv2D, UpSampling2D, Input, Lambda, Concatenate, Add, BatchNormalization, \
    MaxPool2D, GlobalAveragePooling2D, Dense, AvgPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.regularizers import l2
from keras.utils import plot_model
import tensorflow as tf


class DConv(object):
    counter = 0

    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 regularizer=None):
        self.op = DepthwiseConv2D(kernel_size=kernel_size,
                                  strides=strides,
                                  padding='same',
                                  depth_multiplier=1,
                                  use_bias=False,
                                  depthwise_regularizer=regularizer,
                                  name=f'dconv_{DConv.counter}')

        DConv.counter += 1

    def __call__(self, tensor):
        with K.name_scope('DConv'):
            return self.op(inputs=tensor)


class GConv(object):
    cgi = 0
    cgc = 0

    def __init__(self,
                 num_groups,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 kernel_regularizer=None):
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.num_ch_per_group = in_channels // num_groups
        self.num_out_ch_per_group = out_channels // num_groups

        self.kernel_size = kernel_size
        self.strides = strides
        self.kernel_regularizer = kernel_regularizer

    def __call__(self, tensor):
        with K.name_scope('GConv'):
            self.conveds = []
            for group in range(self.num_groups):
                x = Lambda(self._group_maker,
                           name=f'group_maker_{GConv.cgi}',
                           arguments={'group': group})(tensor)
                self.conveds.append(self._conv_op()(x))

                GConv.cgi += 1

            if self.num_groups > 1:
                return Concatenate(axis=-1)(self.conveds)
            return self.conveds[0]

    def _conv_op(self):
        op =  Conv2D(filters=self.num_out_ch_per_group,
                     kernel_size=self.kernel_size,
                     strides=self.strides,
                     padding='same',
                     use_bias=False,
                     kernel_regularizer=self.kernel_regularizer,
                     name=f'pointwise_conv_{GConv.cgc}')
        GConv.cgc += 1

        return op

    def _group_maker(self, tensor, group):
        return (tensor[:, :, :,
                group * self.num_ch_per_group:
                (group + 1) * self.num_ch_per_group])


class ShuffleNetUnit(object):
    count_shuffles = 0

    def __init__(self,
                 num_groups,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bottleneck_ratio=4,
                 regularizer=l2(0.01),
                 downsampling=False):
        self.bottleneck_ratio = bottleneck_ratio
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.regularizer = regularizer
        self.downsampling = downsampling
        self.strides = (2, 2) if downsampling else (1, 1)

    def __call__(self, tensor):
        with K.name_scope('ShuffleNetUnit'):
            x = GConv(self.num_groups,
                      self.in_channels,
                      self.in_channels // self.bottleneck_ratio,
                      kernel_regularizer=self.regularizer)(tensor)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.1)(x)

            x = Lambda(self._shuffle_channels, name=f'channel_shuffle_{ShuffleNetUnit.count_shuffles}')(x)
            ShuffleNetUnit.count_shuffles += 1

            x = DConv(self.kernel_size,
                      self.strides,
                      self.regularizer)(x)
            x = BatchNormalization()(x)

            if not self.downsampling:
                x = GConv(self.num_groups,
                          self.in_channels // self.bottleneck_ratio,
                          self.out_channels,
                          kernel_regularizer=self.regularizer)(x)
            else:
                x = GConv(self.num_groups,
                          self.in_channels // self.bottleneck_ratio,
                          self.out_channels - self.in_channels,
                          kernel_regularizer=self.regularizer)(x)

            x = BatchNormalization()(x)

            if not self.downsampling:
                x = Add()([tensor, x])
            else:
                downsampled = AvgPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(tensor)
                x = Concatenate(axis=-1)([downsampled, x])

            return LeakyReLU(0.1)(x)

    def _shuffle_channels(self, tensor):
        _, h, w, c = K.int_shape(tensor)
        num_ch_per_group = c // self.num_groups

        if num_ch_per_group * self.num_groups != c:
            raise ValueError(f'Number of groups ({self.num_groups}) does not evenly divide '
                             f'the number of channels ({c}) in the tensor.')

        x = K.reshape(tensor, (-1, h, w, num_ch_per_group, self.num_groups))
        x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
        x = K.reshape(x, (-1, h, w, c))

        return x


class ShuffleNetStage(object):

    def __init__(self,
                 num_repeat,
                 num_groups,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 bottleneck_ratio=4):
        self.num_repeat = num_repeat
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bottleneck_ratio = bottleneck_ratio

    def __call__(self, tensor):
        with K.name_scope('Shufflenet_Stage'):
            x = self._shuffle_net_downsampling_op()(tensor)

            for _ in range(self.num_repeat):
                x = self._shuffle_net_regular_op()(x)

            return x

    def _shuffle_net_regular_op(self):
        return ShuffleNetUnit(self.num_groups,
                              self.out_channels,
                              self.out_channels,
                              self.kernel_size,
                              self.bottleneck_ratio)

    def _shuffle_net_downsampling_op(self):
        return ShuffleNetUnit(self.num_groups,
                              self.in_channels,
                              self.out_channels,
                              self.kernel_size,
                              self.bottleneck_ratio,
                              downsampling=True)


class GlobalGammaPooling2D(object):

    def __init__(self, gamma, mode='exp'):
        self.gamma = gamma

        if mode not in ['exp', 'harmonic']:
            raise ValueError('Mode can only be one of `exp` and `harmonic`.')

        if mode == 'exp':
            self.func = self._exp_gamma_pooling
        else:
            self.func = self._harmonic_gamma_pooling

    def __call__(self, tensor):
        with K.name_scope('Global_Gamma_Pooling'):
            return Lambda(self.func)(tensor)

    def _exp_gamma_pooling(self, tensor):
        with K.name_scope('Exponential_Mode'):
            _, h, w, c = K.int_shape(tensor)
            pool_size = h * w

            gamma_tensor = tf.constant([self.gamma ** i for i in range(pool_size)])

            x = K.reshape(tensor, (-1, pool_size, c))
            x = K.permute_dimensions(x, (0, 2, 1))
            x = tf.nn.top_k(x, k=pool_size, sorted=True)[0]
            x = tf.multiply(x, gamma_tensor)
            x = tf.reduce_sum(x, axis=-1)

            return x

    def _harmonic_gamma_pooling(self, tensor):
        with K.name_scope('Harmonic_Mode'):
            _, h, w, c = K.int_shape(tensor)
            pool_size = h * w

            gamma_tensor = tf.constant([self.gamma / i for i in range(1, pool_size + 1)])

            x = K.reshape(tensor, (-1, pool_size, c))
            x = K.permute_dimensions(x, (0, 2, 1))
            x = tf.nn.top_k(x, k=pool_size, sorted=True)[0]
            x = tf.multiply(x, gamma_tensor)
            x = tf.reduce_sum(x, axis=-1)

            return x


class SamplingAndFlow(object):

    def __init__(self):
        raise NotImplementedError('Implement a Flow!')


if __name__ == '__main__':
    from keras.datasets import cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    in_img = Input(shape=(32, 32, 3))

    with K.name_scope('Simple_Convolution'):
        # x = Conv2D(24, (3, 3), strides=(2, 2), padding='same')(in_img)
        x = Conv2D(24, (3, 3), strides=(1, 1), padding='same')(in_img)  # for CIFAR-10
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        # x = MaxPool2D((3, 3), (2, 2), 'same')(x)  # not using for CIFAR-10

    x = ShuffleNetStage(3, 4, 24, 120)(x)
    x = ShuffleNetStage(7, 4, 120, 240)(x)
    x = ShuffleNetStage(3, 4, 240, 480)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)

    # x = ShuffleNetUnit(1, 2, 8,
    #                    kernel_size=(5, 5),
    #                    bottleneck_ratio=1,
    #                    downsampling=True)(in_sound)
    #
    # # x = GConv(1, 2, 8,
    # #           kernel_size=(5, 5),
    # #           strides=(2, 2),
    # #           kernel_regularizer=l2(0.01))(in_sound)
    #
    # x = ShuffleNetUnit(1, 8, 32,
    #                    kernel_size=(5, 5),
    #                    bottleneck_ratio=1,
    #                    downsampling=True)(x)
    #
    # x = ShuffleNetUnit(2, 32, 64,
    #                    kernel_size=(5, 5),
    #                    bottleneck_ratio=2,
    #                    downsampling=True)(x)
    #
    # x = ShuffleNetUnit(4, 64, 128,
    #                    kernel_size=(5, 5),
    #                    bottleneck_ratio=4,
    #                    downsampling=True)(x)
    #
    # x = ShuffleNetUnit(8, 128, 128,
    #                    kernel_size=(3, 3),
    #                    bottleneck_ratio=4,
    #                    downsampling=False)(x)
    #
    # x = ShuffleNetUnit(8, 128, 128,
    #                    kernel_size=(3, 3),
    #                    bottleneck_ratio=4,
    #                    downsampling=False)(x)

    model = Model(in_img, x)
    model.summary()
    # plot_model(model, '../model_vis/prototype_shufflenet_model.png', show_shapes=True)
    tensorboard = TensorBoard(log_dir='../cifar10_logs')
    model.compile(optimizer='sgd', loss='mse')
    try:
        model.fit(steps_per_epoch=1, callbacks=[tensorboard])
    except:
        pass

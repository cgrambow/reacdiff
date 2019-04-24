import keras

import reacdiff.models.layers as layers
import reacdiff.utils as utils

# Wrap layers using TimeDistributed
_wrapped_layers = utils.LayersWrapper(keras.layers)


class DenseNet:
    """A densely connected convolutional encoder"""

    def __init__(self,
                 inputs=keras.layers.Input(shape=(128, 128, 1), name='conv_input'),
                 feat_maps=16,
                 first_conv_size=7,
                 first_conv_stride=2,
                 first_pool_size=3,
                 first_pool_stride=2,
                 growth_rate=12,
                 blocks=(3, 4, 5),
                 dropout=0.0,
                 reduction=0.5,
                 bottleneck=True,
                 flatten_last=False,
                 latent_units=100):
        """
        :param inputs: input tensor.
        :param feat_maps: int, number of feature maps in first convolutional layer.
        :param first_conv_size: int, filter size in first convolutional layer.
        :param first_conv_stride: int, strides in first convolutional layer.
        :param first_pool_size: int, window size in first pool.
        :param first_pool_stride: int, strides in first pool.
        :param growth_rate: int, growth rate in dense blocks.
        :param blocks: sequence of numbers of layers in each dense block.
        :param dropout: float, dropout rate after convolutions.
        :param reduction: float, compression rate in transition blocks.
        :param bottleneck: bool, use 1x1 bottleneck convolution in dense blocks.
        :param flatten_last: bool, flatten instead of global pool before output.
        :param latent_units: int, size of final hidden representation.
        """
        self.inputs = inputs
        self.outputs = None

        # First convolution and pool settings
        self.feat_maps = feat_maps
        self.first_conv_size = first_conv_size
        self.first_conv_stride = first_conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride

        # Dense and transition block settings
        self.growth_rate = growth_rate
        self.blocks = blocks
        self.dropout = dropout
        self.reduction = reduction
        self.bottleneck = bottleneck

        # Output settings
        self.flatten_last = flatten_last
        self.latent_units = latent_units

    def build(self):
        # First convolutional and pooling layers
        x = _wrapped_layers.Conv2D(self.feat_maps, self.first_conv_size,
                                   strides=self.first_conv_stride,
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=keras.initializers.he_uniform(),
                                   name='conv')(self.inputs)
        if self.first_pool_size > 0:
            x = _wrapped_layers.BatchNormalization(epsilon=1.001e-5,
                                                   gamma_regularizer=keras.regularizers.l2(1e-4),
                                                   name='bn')(x)
            x = _wrapped_layers.Activation('relu', name='relu')(x)
            x = _wrapped_layers.MaxPooling2D(self.first_pool_size,
                                             strides=self.first_pool_stride,
                                             padding='same',
                                             name='pool')(x)

        # Dense and transition blocks
        for i, b in enumerate(self.blocks):
            x = layers.dense_block(x, b,
                                   growth_rate=self.growth_rate,
                                   dropout=self.dropout,
                                   bottleneck=self.bottleneck,
                                   name='dense' + str(i + 1))
            if i < len(self.blocks) - 1:
                x = layers.transition_block(x,
                                            dropout=self.dropout,
                                            reduction=self.reduction,
                                            name='transition' + str(i + 1))

        # Convert to vector
        if self.flatten_last:
            vec = _wrapped_layers.Flatten(name='flatten')(x)
        else:
            vec = _wrapped_layers.GlobalAveragePooling2D(name='global_pool')(x)

        self.outputs = _wrapped_layers.Dense(self.latent_units,
                                             activation='relu',
                                             kernel_initializer=keras.initializers.he_uniform(),
                                             name='dense')(vec)

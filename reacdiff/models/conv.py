import keras

import reacdiff.models.layers as layers


class DenseNet:
    """A densely connected convolutional encoder"""

    def __init__(self,
                 input_shape=(128, 128, 1),
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
                 flatten_last=False):
        """
        :param input_shape: shape of input tensor.
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
        """
        self.inputs = keras.layers.Input(shape=input_shape)
        self.model = None

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

    def build(self):
        # First convolutional and pooling layers
        x = keras.layers.Conv2D(self.feat_maps, self.first_conv_size,
                                strides=self.first_conv_stride,
                                padding='same',
                                use_bias=False,
                                kernel_initializer=keras.initializers.he_uniform(),
                                name='conv')(self.inputs)
        if self.first_pool_size > 0:
            x = keras.layers.BatchNormalization(axis=layers.bn_axis, epsilon=1.001e-5,
                                                gamma_regularizer=keras.regularizers.l2(1e-4),
                                                name='bn')(x)
            x = keras.layers.Activation('relu', name='relu')(x)
            x = keras.layers.MaxPooling2D(self.first_pool_size,
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
            outputs = keras.layers.Flatten(name='flatten')(x)
        else:
            outputs = keras.layers.GlobalAveragePooling2D(name='global_pool')(x)

        self.model = keras.models.Model(self.inputs, outputs, name='dense_net')

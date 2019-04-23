import keras

import reacdiff.utils as utils

# Wrap layers using TimeDistributed
_wrapped_layers = utils.LayersWrapper(keras.layers)


def dense_block(x, blocks, name='', **kwargs):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x1 = conv_block(x, name=name + '/block' + str(i + 1), **kwargs)
        x = keras.layers.Concatenate(name=name + '/concat' + str(i + 1))([x, x1])
    return x


def transition_block(x, dropout=0.0, reduction=0.5, name='', **kwargs):
    """A downsampling transition block.
    # Arguments
        x: input tensor.
        dropout: float, dropout rate.
        reduction: float, compression rate for filters.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    x = _wrapped_layers.BatchNormalization(epsilon=1.001e-5,
                                           gamma_regularizer=keras.regularizers.l2(1e-4),
                                           name=name + '/bn')(x)
    x = _wrapped_layers.Activation('relu', name=name + '/relu')(x)
    x = _wrapped_layers.Conv2D(int(keras.backend.int_shape(x)[-1] * reduction), 1,
                               use_bias=False,
                               kernel_initializer=keras.initializers.he_uniform(),
                               name=name + '/conv')(x)
    x = _wrapped_layers.Dropout(dropout, name=name + '/dropout')(x)
    x = _wrapped_layers.AveragePooling2D(2, strides=2, name=name + '/pool')(x)
    return x


def conv_block(x, growth_rate=12, dropout=0.0, bottleneck=True, name=''):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        dropout: float, dropout rate.
        bottleneck: bool, use 1x1 bottleneck convolution
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    if bottleneck:
        x = _wrapped_layers.BatchNormalization(epsilon=1.001e-5,
                                               gamma_regularizer=keras.regularizers.l2(1e-4),
                                               name=name + '_0_bn')(x)
        x = _wrapped_layers.Activation('relu', name=name + '_0_relu')(x)
        x = _wrapped_layers.Conv2D(4 * growth_rate, 1,
                                   use_bias=False,
                                   kernel_initializer=keras.initializers.he_uniform(),
                                   name=name + '_0_conv')(x)
        x = _wrapped_layers.Dropout(dropout, name=name + '_0_dropout')(x)
    x = _wrapped_layers.BatchNormalization(epsilon=1.001e-5,
                                           gamma_regularizer=keras.regularizers.l2(1e-4),
                                           name=name + '_bn')(x)
    x = _wrapped_layers.Activation('relu', name=name + '_relu')(x)
    x = _wrapped_layers.Conv2D(growth_rate, 3,
                               padding='same',
                               use_bias=False,
                               kernel_initializer=keras.initializers.he_uniform(),
                               name=name + '_conv')(x)
    x = _wrapped_layers.Dropout(dropout, name=name + '_dropout')(x)
    return x

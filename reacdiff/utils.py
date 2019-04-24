import functools

import keras
import keras.backend as backend
import numpy as np


def shuffle_arrays(*args, seed=None):
    n = len(args[0])
    assert all(len(a) == n for a in args)
    p = np.random.RandomState(seed=seed).permutation(n)
    return tuple(a[p] for a in args)


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true)))


def mae(y_true, y_pred):
    return backend.mean(backend.abs(y_pred - y_true))


def tdist(obj):
    @functools.wraps(obj)
    def wrapper(*args, **kwargs):
        return keras.layers.TimeDistributed(obj(*args, **kwargs), name=kwargs.get('name'))
    return wrapper


class LayersWrapper:
    """Class for wrapping some layers from a module in TimeDistributed."""

    layer_names = {'BatchNormalization',
                   'Activation',
                   'Conv2D',
                   'Dropout',
                   'AveragePooling2D',
                   'MaxPooling2D',
                   'Flatten',
                   'GlobalAveragePooling2D',
                   'Dense'}

    def __init__(self, module):
        for name in self.layer_names:
            wrapped_layer = tdist(getattr(module, name))
            setattr(self, name, wrapped_layer)

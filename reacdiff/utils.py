import os

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

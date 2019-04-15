import os

import h5py
import numpy as np

import reacdiff.data.data as datamod


def preprocess(path):
    """
    :param path: An HDF5 file.
    """
    f = h5py.File(path, 'r')
    targets = f['A1']
    states = f['y']

    assert len(targets.shape) == 2
    assert len(states.shape) == 5
    assert states.shape[0] == 2

    targets = targets[:]
    states = states[:]
    targets = np.swapaxes(targets, 0, 1)
    states = np.transpose(states, (0, 4, 1, 2, 3))
    data, data2 = np.expand_dims(states, axis=-1)

    name = os.path.splitext(path)[0]
    targets_path = name + '_targets.h5'
    data_path = name + '_states.h5'
    data_path2 = name + '_states2.h5'

    datamod.save_data(targets, targets_path)
    datamod.save_data(data, data_path)
    datamod.save_data(data2, data_path2)

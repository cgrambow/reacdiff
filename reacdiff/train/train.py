import os

import keras

import reacdiff.utils as utils


def train(crnn, data, save_dir,
          batch_size=32,
          epochs=30,
          val_data=None,
          patience=5,
          lr_start=1e-3,
          lr_end=1e-4,
          max_norm=2.0,
          quiet=False):
    """
    :param crnn: CRNN instance.
    :param data: Training data.
    :param save_dir: directory to save models to.
    :param batch_size: int, batch size.
    :param epochs: maximum number of epochs.
    :param val_data: Validation data.
    :param patience: patience.
    :param lr_start: float, initial learning rate.
    :param lr_end: float, final learning rate.
    :param max_norm: float, maximum gradient norm.
    :param quiet: bool, print less information
    """
    optimizer = keras.optimizers.Adam(lr=lr_start, clipnorm=max_norm)
    crnn.model.compile(optimizer=optimizer, loss='mse', metrics=[utils.rmse, utils.mae])
    if not quiet:
        crnn.encoder.summary()
        crnn.rnn.summary()
        crnn.model.summary()

    model_name = 'model.{epoch:03d}.h5'
    model_path = os.path.join(save_dir, model_name)

    def lr_schedule(epoch, _):
        return lr_start * ((lr_end / lr_start) ** (epoch / epochs))

    checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                 verbose=1,
                                                 save_best_only=True)
    early_stopping = keras.callbacks.EarlyStopping(patience=patience,
                                                   verbose=1,
                                                   restore_best_weights=True)
    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule,
                                                         verbose=1)
    callbacks = [checkpoint, early_stopping, lr_scheduler]

    crnn.model.fit(data.get_data(), data.targets,
                   batch_size=batch_size,
                   epochs=epochs,
                   validation_data=(val_data.get_data(), val_data.targets),
                   shuffle=True,
                   verbose=2 if quiet else 1,
                   callbacks=callbacks)

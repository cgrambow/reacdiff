import os

import keras

import reacdiff.data.data as datamod
import reacdiff.utils as utils


def predict(args):
    # Load data
    print('Loading data')
    data = datamod.Dataset(
        datamod.load_data(args.data_path),
        data2=None if args.data_path2 is None else datamod.load_data(args.data_path2)
    )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Load model
    model = keras.models.load_model(args.model, custom_objects={'rmse': utils.rmse, 'mae': utils.mae})

    # Predict
    preds = model.predict(data.get_data(), batch_size=args.batch_size, verbose=1)
    datamod.save_csv(preds, args.save_path)

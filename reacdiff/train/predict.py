import glob
import os
import re

import keras
import numpy as np

import reacdiff.data.data as datamod
import reacdiff.utils as utils


def predict(args):
    # Load data
    print('Loading data')
    data = datamod.Dataset(
        datamod.load_data(args.data_path),
        targets=None if args.targets_path is None else datamod.load_csv(args.targets_path),
        data2=None if args.data_path2 is None else datamod.load_data(args.data_path2)
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    # Walk directory for ensemble of models
    if os.path.isdir(args.model):
        model_dirs = glob.iglob(os.path.join(args.model, 'model*'))
        model_nums = [int(re.search('\d+', os.path.basename(d))[0]) for d in model_dirs]
        model_nums.sort()

        targets_size = args.targets_size if args.targets_path is None else data.targets.shape[1]
        all_preds = np.zeros((len(model_nums), len(data), targets_size))

        for model_idx in model_nums:
            print(f'Evaluating model {model_idx}')

            model_path = os.path.join(args.model, f'model{model_idx}', 'model.h5')
            model = keras.models.load_model(model_path, custom_objects={'rmse': utils.rmse, 'mae': utils.mae})

            preds = model.predict(data.get_data(), batch_size=args.batch_size, verbose=1)
            all_preds[model_idx] = preds
        preds = np.mean(all_preds, axis=0)
    else:
        # Load model
        model = keras.models.load_model(args.model, custom_objects={'rmse': utils.rmse, 'mae': utils.mae})

        # Predict
        preds = model.predict(data.get_data(), batch_size=args.batch_size, verbose=1)

    if args.targets_path is not None:
        print('Evaluating ensemble')
        rmse = utils.rmse_np(data.targets, preds)
        mae = utils.mae_np(data.targets, preds)
        print(f'rmse: {rmse:.4f}; mae: {mae:.4f}')
    datamod.save_csv(preds, args.save_path)

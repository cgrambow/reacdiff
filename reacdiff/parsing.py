import argparse
import os


def parse_dataprep_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to HDF5 data file')
    return parser.parse_args()


def parse_predict_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data containing states for prediction task')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--data_path2', type=str,
                        help='Path to additional observable states for prediction')
    parser.add_argument('--save_path', type=str, default=os.path.join(os.getcwd(), 'preds.csv'),
                        help='Path to save predictions to')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    return parser.parse_args()


def parse_train_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data containing observable states')
    parser.add_argument('--targets_path', type=str, required=True,
                        help='Path to targets')
    parser.add_argument('--data_path2', type=str,
                        help='Path to data containing additional observable states')
    parser.add_argument('--save_dir', type=str, default=os.getcwd(),
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--noise_frac', type=float, default=0.0,
                        help='Fraction of standard deviation of data to add as noise')
    parser.add_argument('--splits', type=float, nargs=3, default=[0.9, 0.05, 0.05],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--save_test_data', action='store_true',
                        help='Save test data and targets to file')
    parser.add_argument('--test_data_path', type=str,
                        help='Path to separate test data')
    parser.add_argument('--test_targets_path', type=str,
                        help='Path to separate test targets')
    parser.add_argument('--test_data_path2', type=str,
                        help='Path to separate test data for additional observable')
    parser.add_argument('--quiet', action='store_true',
                        help='Do not print model details and batch information during training')
    parser.add_argument('--cpu', action='store_true',
                        help='Run on CPU instead of GPU')
    parser.add_argument('--seed', type=int, default=7,
                        help='Random seed for reproducibility')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Maximum number of epochs to run')
    parser.add_argument('--lr_start', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--lr_end', type=float, default=1e-4,
                        help='Final learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs without improvement after which training will be stopped')
    parser.add_argument('--max_norm', type=float, default=2.0,
                        help='Maximum gradient norm before clipping occurs')

    # Encoder arguments
    parser.add_argument('--feat_maps', type=int, default=16,
                        help='Number of feature maps in first convolutional layer')
    parser.add_argument('--first_conv_size', type=int, default=7,
                        help='Filter size in first convolutional layer')
    parser.add_argument('--first_conv_stride', type=int, default=2,
                        help='Strides in first convolutional layer')
    parser.add_argument('--first_pool_size', type=int, default=3,
                        help='Window size in first pool')
    parser.add_argument('--first_pool_stride', type=int, default=2,
                        help='Strides in first pool')
    parser.add_argument('--growth_rate', type=int, default=12,
                        help='Growth rate in dense blocks')
    parser.add_argument('--blocks', type=int, nargs='+', default=[3, 4, 5],
                        help='Numbers of layers in each dense block')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate after convolutions')
    parser.add_argument('--reduction', type=float, default=0.5,
                        help='Compression rate in transition blocks')
    parser.add_argument('--no_bottleneck', action='store_true',
                        help='Do not use bottleneck convolution in dense blocks')
    parser.add_argument('--flatten_last', action='store_true',
                        help='Flatten instead of global pool before output')
    parser.add_argument('--latent_units', type=int, default=100,
                        help='Number of units in latent representation prior to RNN')

    # RNN arguments
    parser.add_argument('--rnn_layers', type=int, default=1,
                        help='Number of RNN layers')
    parser.add_argument('--rnn_units', type=int, default=100,
                        help='Number of units in RNN')

    return parser.parse_args()

import os

import reacdiff.data.data as datamod
import reacdiff.models.crnn as models
import reacdiff.train.train as train


def run_training(args):
    # Load data
    print('Loading data')
    data = datamod.Dataset(
        datamod.load_data(args.data_path),
        datamod.load_data(args.targets_path),
        None if args.data_path2 is None else datamod.load_data(args.data_path2)
    )

    # Split data
    print(f'Splitting data with seed {args.seed}')
    train_data, val_data, test_data = datamod.split_data(data, splits=args.splits, seed=args.seed)

    if args.test_data_path is not None and args.test_targets_path is not None:
        test_data = datamod.Dataset(
            datamod.load_data(args.test_data_path),
            datamod.load_data(args.test_targets_path),
            None if args.test_data_path2 is None else datamod.load_data(args.test_data_path2)
        )

    os.makedirs(args.save_dir, exist_ok=True)

    if args.save_test_data:
        test_data.save(os.path.join(args.save_dir, 'test'))

    print(f'Train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Build model
    crnn = models.CRNN(
        time_steps=train_data.data.shape[1],
        input_shape=train_data.data.shape[2:],
        output_dim=train_data.targets.shape[1],
        observables=train_data.get_num_observables(),
        rnn_layers=args.rnn_layers,
        rnn_units=args.rnn_units,
        use_gpu=not args.cpu
    )
    crnn.build(
        feat_maps=args.feat_maps,
        first_conv_size=args.first_conv_size,
        first_conv_stride=args.first_conv_stride,
        first_pool_size=args.first_pool_size,
        first_pool_stride=args.first_pool_stride,
        growth_rate=args.growth_rate,
        blocks=args.blocks,
        dropout=args.dropout,
        reduction=args.reduction,
        bottleneck=not args.no_bottleneck,
        flatten_last=args.flatten_last
    )

    model_dir = os.path.join(args.save_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    # Train model
    train.train(crnn, train_data, model_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                val_data=val_data,
                patience=args.patience,
                lr_start=args.lr_start,
                lr_end=args.lr_end,
                max_norm=args.max_norm,
                quiet=args.quiet)

    # Evaluate on test set
    print('Evaluating test data')
    metrics = crnn.model.evaluate(test_data.get_data(), test_data.targets,
                                  batch_size=args.batch_size)
    print(*(f'{name}: {metric:.4f}' for name, metric in zip(crnn.model.metrics_names, metrics)), sep='; ')

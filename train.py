#!/usr/bin/env python

import reacdiff.parsing as parsing
import reacdiff.train.run_training as run_training


if __name__ == '__main__':
    args = parsing.parse_train_args()
    run_training.run_training(args)

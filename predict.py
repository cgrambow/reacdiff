#!/usr/bin/env python

import reacdiff.parsing as parsing
import reacdiff.train.predict as predict


if __name__ == '__main__':
    args = parsing.parse_predict_args()
    predict.predict(args)

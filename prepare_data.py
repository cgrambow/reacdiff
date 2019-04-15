#!/usr/bin/env python

import reacdiff.parsing as parsing
import reacdiff.data.preprocess as preprocess


if __name__ == '__main__':
    args = parsing.parse_dataprep_args()
    preprocess.preprocess(args.data_path)

import numpy as np
import sys


def parse_args_with_help(parser):
    try:
        args = parser.parse_args()
        return args
    except:
        parser.print_help()
        sys.exit(0)


def get_values_from_file(path, delimiter):
    values = []
    for l in open(path):
        parts = l.strip().split(delimiter)
        values.append(float(parts[0]))
    return values


def read_feature_files(fpaths, scale=False):
    features = []
    for fpath in fpaths:
        values = get_values_from_file(fpath, delimiter=' ||| ')
        values = np.asarray(values)
        if scale:
            values = preprocessing.scale(values)
        features.append(values)
    return np.stack(features, axis=-1)


def read_labels_file(fpath):
    return np.loadtxt(fpath)


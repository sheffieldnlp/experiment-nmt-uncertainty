import argparse
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from utils import parse_args_with_help
from utils import get_values_from_file


def get_classifier(clf_name):
    if clf_name == 'SVR':
        return SVR
    elif clf_name == 'LinearSVR':
        return LinearSVR
    elif clf_name == 'MLPRegressor':
        return MLPRegressor
    elif clf_name == 'GradientBoostingRegressor':
        return GradientBoostingRegressor
    elif clf_name == 'GaussianProcessRegressor':
        return GaussianProcessRegressor
    elif clf_name == 'LinearRegression':
        return linear_model.LinearRegression()
    else:
        print('Error! Unknown classifier')


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


def correlation_fn(y_true, y_pred, measure='Pearson'):
    if measure == 'Pearson':
        return pearsonr(y_true, y_pred)[0]
    elif measure == 'Spearman':
        return spearmanr(y_true, y_pred)[0]
    else:
        print('Error! Unknown measure')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_files', nargs='+')
    parser.add_argument('--labels_file')
    parser.add_argument('-m', '--measure', required=False, default='Spearman')
    parser.add_argument('-c', '--classifier', required=False, default='SVR')
    parser.add_argument('-s', '--scale', action='store_true', required=False, default=False)
    args = parse_args_with_help(parser)
    x_data = read_feature_files(args.feature_files, scale=args.scale)
    y_data = read_labels_file(args.labels_file)
    clf = get_classifier(args.classifier)()
    scorer = make_scorer(correlation_fn, measure=args.measure)
    scores = cross_val_score(clf, x_data, y_data, scoring=scorer, cv=5)
    print(scores)
    print('Correlation: %0.3f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()

import argparse

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from utils import parse_args_with_help
from utils import read_feature_files
from utils import read_labels_file


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


def correlation_fn(y_true, y_pred, measure='Pearson'):
    if measure == 'Pearson':
        return pearsonr(y_true, y_pred)[0]
    elif measure == 'Spearman':
        return spearmanr(y_true, y_pred)[0]
    else:
        print('Error! Unknown measure')


def cross_validation(feature_files, labels_file, measure='Spearman', classifier='SVR', scale=False):
    x_data = read_feature_files(feature_files, scale=scale)
    y_data = read_labels_file(labels_file)
    clf = get_classifier(classifier)()
    scorer = make_scorer(correlation_fn, measure=measure)
    scores = cross_val_score(clf, x_data, y_data, scoring=scorer, cv=5)
    print(scores)
    print('Correlation: %0.3f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_files', nargs='+')
    parser.add_argument('--labels_file')
    parser.add_argument('-m', '--measure', required=False, default='Spearman')
    parser.add_argument('-c', '--classifier', required=False, default='SVR')
    parser.add_argument('-s', '--scale', action='store_true', required=False, default=False)
    args = parse_args_with_help(parser)
    cross_validation(args.feature_files, args.labels_file, measure=args.measure, classifier=args.classifier, scale=args.scale)


if __name__ == '__main__':
    main()


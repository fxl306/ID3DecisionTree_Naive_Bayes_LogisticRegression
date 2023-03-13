"""
 This program implements the naïve bayes algorithm,
 the range of the feature will be partitioned into *k* bins (value set through an option).
 The m-estimates is used to smooth the probability estimates.
 The logs is used whenever possible to avoid multiplying too many probabilities together.


 @Author Feng Long: fxl306@case.edu
 @Date 11/04/2022
 """

import argparse
import sys
from cmath import log
import os.path
import warnings

from typing import Optional, List

import numpy as np
from sting.classifier import Classifier
from sting.data import Feature, parse_c45, FeatureType
import math

import util


class NaiveBayes(Classifier):
    def __init__(self, schema: List[Feature], num_bins: int, m: float):

        self._schema = schema  # For some models (like a decision tree) it makes sense to keep track of the data schema
        self._num_bins = num_bins
        self._m = m
        self._prior = 0
        self._params = []
        self.cont_feat = []
        self.cont_feat_p = []

        for f in schema:
            self.cont_feat.append([])
            self.cont_feat_p.append([])

    def fit(self, X: np.ndarray, y: np.ndarray, bins, weights: Optional[np.ndarray] = None) -> None:
        """
        This is the method where the training algorithm will run.
        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            weights: Weights for each example. Will become relevant later in the course, ignore for now.
        """

        # In Java, it is best practice to LBYL (Look Before You Leap), i.e. check to see if code will throw an exception
        # BEFORE running it. In Python, the dominant paradigm is EAFP (Easier to Ask Forgiveness than Permission), where
        # try/except blocks (like try/catch blocks) are commonly used to catch expected exceptions and deal with them.
        # try:
        # split_criterion = self._determine_split_criterion(X, y)
        # except NotImplementedError:
        # warnings.warn('This is for demonstration purposes only.')

        class_zero_examples, class_one_examples = self.split(X, y)
        self.prior = len(class_one_examples) / len(X)

        for i in range(len(self.schema)):
            if self.schema[i].ftype == FeatureType.BINARY:
                self.params.append(self._params_binary(i, class_zero_examples, class_one_examples))
            elif self.schema[i].ftype == FeatureType.NOMINAL:
                self.params.append(self._params_nominal(i, class_zero_examples, class_one_examples))
            else:
                self.params.append(self._params_continuous(i, class_zero_examples, class_one_examples, bins))

    def split(self, X: np.ndarray, y: np.ndarray):
        class_zero_examples = []
        class_one_examples = []
        for i in range(len(X)):
            if y[i] == 0:
                class_zero_examples.append(X[i])
            else:
                class_one_examples.append(X[i])
        return class_zero_examples, class_one_examples

    def _params_binary(self, feature_index: int, class_zero_examples: List[np.ndarray],
                       class_one_examples: List[np.ndarray]):
        p = 1 / 2
        m_i = self.m
        if self.m < 0:
            m_i = 2

        params = []
        for x_class in [class_zero_examples, class_one_examples]:
            num_Xi_being_1 = 0
            for x in x_class:
                if x[feature_index] == 1:
                    num_Xi_being_1 += 1
            params.append((num_Xi_being_1 + m_i * p) / (len(x_class) + m_i))

        # params = [P(xi = true|y = 0), P(xi = true|y = 1)]
        return params

    def _params_nominal(self, feature_index: int, class_zero_examples: List[np.ndarray],
                        class_one_examples: List[np.ndarray]):
        v = len(self.schema[feature_index].values)
        p = 1 / v
        m_i = self.m
        if self.m < 0:
            m_i = v
        params = []
        for x_class in [class_zero_examples, class_one_examples]:
            num_of_each_value = np.zeros(v, dtype=int)
            for x in x_class:
                for val_i in range(v):
                    if x[feature_index] == val_i + 1:
                        num_of_each_value[val_i] += 1
            params_of_class = []
            for i in range(v):
                params_of_class.append((num_of_each_value[i] + m_i * p) / (len(x_class) + m_i))
            params.append(params_of_class)

        # params = [[P(xi = v1|y = 0), P(xi = v2|y = 0), ...] [P(xi = v1|y = 1), P(xi = v2|y = 1), ...]]
        return params

    def _params_continuous(self, feature_index: int, class_zero_examples: List[np.ndarray],
                           class_one_examples: List[np.ndarray], bin):
        # implement continuous here
        minv: float = 100000
        maxv: float = -1
        xlist = [[], []]

        # feature_index = 0
        # class_one_examples = [[1.0],[5.0],[6.0],[9.0]]
        # class_zero_examples = [[4.0],[5.0],[7.0],[19.0]]

        for x in class_zero_examples:
            xlist[0].append(x[feature_index])
            minv = min(x[feature_index], minv)
            maxv = max(x[feature_index], maxv)

        for x in class_one_examples:
            xlist[1].append(x[feature_index])
            minv = min(x[feature_index], minv)
            maxv = max(x[feature_index], maxv)

        xlist[0].sort()
        xlist[1].sort()

        bac: float = (maxv - minv) / bin
        start: float = minv
        buk = [[], []]

        while start <= maxv:
            buk[0].append([0])
            buk[1].append([0])

            self.cont_feat[feature_index].append(start)
            start = start + bac

        buk[0].append([0])
        buk[1].append([0])

        self.cont_feat[feature_index][len(self.cont_feat[feature_index]) - 1] += 1

        ind = 0
        for xi in range(len(xlist[0])):
            while ind < len(self.cont_feat[feature_index]) and xlist[0][xi] >= self.cont_feat[feature_index][ind]:
                ind += 1

            buk[0][ind][0] += 1

        ind = 0
        for xi in range(len(xlist[1])):
            while ind < len(self.cont_feat[feature_index]) and xlist[1][xi] >= self.cont_feat[feature_index][ind]:
                ind += 1

            buk[1][ind][0] += 1

        m_i = self.m
        p = 1 / (len(self.cont_feat[feature_index]) - 1)
        pbuk = [[], []]

        if self.m < 0:
            m_i = (len(self.cont_feat[feature_index]) - 1)

        for x in buk[0]:
            pbuk[0].append((x[0] + m_i * p) / (len(class_zero_examples) + m_i))

        for x in buk[1]:
            pbuk[1].append((x[0] + m_i * p) / (len(class_one_examples) + m_i))

        # print(self.cont_feat[0])
        # print(buk)
        # print(pbuk)

        self.cont_feat_p[feature_index] = pbuk

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        This is the method where the naive Bayes is evaluated.
        Args:
            X: The testing data of shape (n_examples, n_features).
        Returns: Predictions of shape (n_examples,), either 0 or 1
        """

        # Returns either all 1s or all 0s, depending on _majority_label.
        # return np.ones(X.shape[0], dtype=int) * self._majority_label
        y_pred = []
        y_conf = []
        for x in X:
            log_P_x_and_h_for_hs = [0, 0]
            for h in [0, 1]:
                if h == 1:
                    log_P_x_and_h = math.log(self.prior)
                else:
                    log_P_x_and_h = math.log(1 - self.prior)
                for feat_i in range(len(self.schema)):
                    if self.schema[feat_i].ftype == FeatureType.BINARY:
                        log_P_x_and_h += math.log(self._get_params_binary(feat_i, x, h))
                    elif self.schema[feat_i].ftype == FeatureType.NOMINAL:
                        log_P_x_and_h += math.log(self._get_params_nominal(feat_i, x, h))
                    else:
                        log_P_x_and_h += math.log(self._get_params_countinuous(feat_i, x, h))
                log_P_x_and_h_for_hs[h] = log_P_x_and_h
            if log_P_x_and_h_for_hs[0] > log_P_x_and_h_for_hs[1]:
                y_pred.append(0)
            else:
                y_pred.append(1)
            y_conf.append(math.exp(log_P_x_and_h_for_hs[1]) / (math.exp(
                log_P_x_and_h_for_hs[0] + math.exp(log_P_x_and_h_for_hs[1])) + 0.001))
        return np.array(y_pred), np.array(y_conf)

    def _get_params_binary(self, feature_index: int, x: np.ndarray, h: int):
        if x[feature_index] == 1:
            return self.params[feature_index][h]
        else:
            return 1 - self.params[feature_index][h]

    def _get_params_nominal(self, feature_index: int, x: np.ndarray, h: int):
        for val_i in range(len(self.schema[feature_index].values)):
            if x[feature_index] == val_i + 1:
                return self.params[feature_index][h][val_i]

    def _get_params_countinuous(self, feature_index: int, x: np.ndarray, h: int):
        for i in range(len(self.cont_feat[feature_index])):
            if x[feature_index] < self.cont_feat[feature_index][i]:
                return self.cont_feat_p[feature_index][h][i]

        return self.cont_feat_p[feature_index][h][len(self.cont_feat[feature_index])]

    # In Python, instead of getters and setters we have properties: docs.python.org/3/library/functions.html#property
    @property
    def schema(self):
        """
        Returns: The dataset schema
        """
        return self._schema

    @property
    def num_bins(self):
        return self._num_bins

    @property
    def m(self):
        return self._m

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        self._prior = value

    @property
    def params(self):
        return self._params


class Printable:
    printInit: bool 

    def __init__(self) -> None:
        self.printInit = False 

    def printOverallMetric(self, modelName: str, accumulate: np.ndarray) -> None:
        if self.printInit is False:
            print("________________________________________________________________\n"
                  "|   Model    |  Accuracy  |  Precision |   Recall   |     AUC    |") 
            self.printInit = True 
        print("-----------------------------------------------------------------")
        print("| {:10s} | {:10.4f} ".format(modelName, accumulate[0]), end="")
        print("| {:10.4f} ".format(accumulate[1]), end="")
        print("| {:10.4f} ".format(accumulate[2]), end="")
        print("| {:10.4f} |".format(accumulate[3]))
        print("-----------------------------------------------------------------")


printable: Printable = Printable() 


def nbayes(data_path: str, num_bins: int, m: int, use_cross_validation: bool = True):
    """
    It is highly recommended that you make a function like this to run your program so that you are able to run it
    easily from a Jupyter notebook. This function has been PARTIALLY implemented for you, but not completely!
    :param data_path: The path to the data.
    :param num_bins: Number of bins for continuous feature
    :param m: An integer for m-estimate. If value is negative, use Laplace smoothing.
    :param use_cross_validation: If True, use cross validation. Otherwise, run on the full dataset.
    :return:
    """

    # last entry in the data_path is the file base (name of the dataset)
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
    root_dir = os.sep.join(path[:-1])
    schema, X, y = parse_c45(file_base, root_dir)

    if use_cross_validation:
        datasets = util.cv_split(X, y, num_folds=5, stratified=True)
    else:
        datasets = ((X, y, X, y),)

    accumulate: np.ndarray = np.zeros(4) 

    allTrueY: np.ndarray = None 
    allYHat: np.ndarray = None 
    for X_train, y_train, X_test, y_test in datasets:
        nbayes = NaiveBayes(schema, num_bins, m)
        nbayes.fit(X_train, y_train, num_bins)
        metrics: np.ndarray = np.array(evaluate_and_print_metrics(nbayes, X_test, y_test)) 
        print() 
        yHat, _ = nbayes.predict(X_test) 
        if allTrueY is None:
            allTrueY = y_test 
            allYHat = yHat 
        else:
            allTrueY = np.concatenate((allTrueY, y_test)) 
            allYHat = np.concatenate((allYHat, yHat)) 

        accumulate += metrics 
    accumulate /= len(datasets) 
    printable.printOverallMetric("Bayes", accumulate) 
    util.printROC(allTrueY, allYHat) 


def evaluate_and_print_metrics(model: Classifier, X: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """
    You will implement this method.
    Given a trained naive Bayes and labelled dataset, Evaluate the naive Bayes and print metrics.
    """

    y_hat, y_conf = model.predict(X)
    acc = util.accuracy(y, y_hat)
    print(f'Accuracy:{acc:.2f}')
    prec = util.precision(y, y_hat)
    print(f'Precision:{prec:.2f}')
    rec = util.recall(y, y_hat)
    print(f'Recall:{rec:.2f}')
    a = util.auc(y, y_conf)
    print(f'Area under ROC:{a:.2f}')

    return (acc, prec, rec, a) 


if __name__ == '__main__':
    """
    THIS IS YOUR MAIN FUNCTION. You will implement the evaluation of the program here. We have provided argparse code
    for you for this assignment, but in the future you may be responsible for doing this yourself.
    """

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a naive Bayes algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('num_bins', type=int,
                        help='Number of bins for any continuous feature. Must be a positive integer(at least 2).')
    parser.add_argument('m', type=float,
                        help='A nonnegative integer for the m-estimate. A negative value means using Laplace smoothing.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.set_defaults(cv=True)
    args = parser.parse_args()

    if args.num_bins < 2:
        raise argparse.ArgumentTypeError('Number of bins must be at least 2.')

    # You can access args with the dot operator like so:
    data_path = os.path.expanduser(args.path)
    use_cross_validation = args.cv

    nbayes(data_path, args.num_bins, args.m, use_cross_validation)
"""
 This implemented the Logistic Regression algorithm.
 During learning, it will minimize the negative conditional log likelihood plus a constant (λ) times a penalty term, half of the 2-norm of the weights squared.
 The standard gradient descent is used for the minimization.
 Nominal attributes are encoded as 1-of-N vectors.

 @Author Feng Long: fxl306@case.edu
 @Date 11/04/2022
 """

import argparse
import os.path
import warnings

from typing import Optional, List

import numpy as np
from sting.data import Feature, parse_c45

import util


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


# In Python, the convention for class names is CamelCase, just like Java! However, the convention for method and
# variable names is lowercase_separated_by_underscores, unlike Java.
class Logreg:
    def __init__(self):
        print()
        #self._schema = schema  # For some models (like a decision tree) it makes sense to keep track of the data schema


    def predict(self, X: np.ndarray, weight, b):
        res = np.asarray(self.sigmoid(X * weight, b)).reshape(-1)
        y_hat = np.zeros(len(res))

        for i in range(len(res)):
            if res[i] >= 1/2:
                y_hat[i] = 1
            else:
                y_hat[i] = 0

        return y_hat, res

    def sigmoid(self, z, b):
        return 1 / (1 + np.exp(-(z + b)))

    def train(self, X, y, b):
        learning_rate = 0.01  # step
        penalty = 0.1

        X = np.asmatrix(X)
        y = np.asmatrix(y).transpose()
        m, n = np.shape(X)
        # init
        weights = np.ones((n, 1))
        grad = (-X.transpose() * (y - self.sigmoid(X * weights, b)) + penalty * weights)  # negative log
        iterations = 4000 # max loop
        iteration = 0

        while iteration < iterations:
            y_hat = self.sigmoid(X * weights, b)
            grad = (-X.transpose() * (y - y_hat) + penalty * weights) # negative log
            weights = weights - learning_rate * grad
            iteration += 1

        return weights


def evaluate(lr: Logreg, X: np.ndarray, y: np.ndarray, weight, b):
    """
    You will implement this method.
    Given a trained decision tree and labelled dataset, Evaluate the tree and print metrics.
    """

    y_hat, y_conf = lr.predict(X, weight, b)
    acc = util.accuracy(y, y_hat)
    print(f'Accuracy:{acc:.2f}')
    prec = util.precision(y, y_hat)
    print(f'Precision:{prec:.2f}')
    rec = util.recall(y, y_hat)
    print(f'Recall:{rec:.2f}')
    a = util.auc(y, y_conf)
    print(f'Area under ROC:{a:.2f}')

    return (acc, prec, rec, a)


def logreg(data_path: str, constant: float, use_cross_validation: bool = True):
    """
    It is highly recommended that you make a function like this to run your program so that you are able to run it
    easily from a Jupyter notebook. This function has been PARTIALLY implemented for you, but not completely!

    :param data_path: The path to the data.
    :param use_cross_validation: If True, use cross validation. Otherwise, run on the full dataset.
    :return:
    """

    # last entry in the data_path is the file base (name of the dataset)
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
    root_dir = os.sep.join(path[:-1])
    schema, X, y = parse_c45(file_base, root_dir)

    if use_cross_validation:
        datasets = util.cv_split(X, y, 5, stratified=True)
    else:
        datasets = ((X, y, X, y),)

    allTrueY: np.ndarray = None 
    allYHat: np.ndarray = None 
    accumulate: np.ndarray = np.zeros(4, dtype=float) 
    for X_train, y_train, X_test, y_test in datasets:
        lr = Logreg()
        weight = lr.train(X_train, y_train, constant)
        ev = evaluate(lr, X_test, y_test, weight, constant)
        accumulate[0] += ev[0]
        accumulate[1] += ev[1]
        accumulate[2] += ev[2]
        accumulate[3] += ev[3]

        yHat,_ = lr.predict(X_test, weight, constant)
        if allTrueY is None:
            allTrueY = y_test 
            allYHat = yHat
        else:
            allTrueY = np.concatenate((allTrueY, y_test)) 
            allYHat = np.concatenate((allYHat, yHat)) 

    accumulate /= len(datasets) 
    printable.printOverallMetric("LogReg", accumulate)
    util.printROC(allTrueY, allYHat) 



if __name__ == '__main__':
    """
    THIS IS YOUR MAIN FUNCTION. You will implement the evaluation of the program here. We have provided argparse code
    for you for this assignment, but in the future you may be responsible for doing this yourself.
    """

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a decision tree algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.set_defaults(cv=True)
    parser.add_argument('const', metavar='CONST', type=float, help='The constant.')

    args = parser.parse_args()

    if args.const < 0:
        raise argparse.ArgumentTypeError('Constant must be nonneg.')

    # You can access args with the dot operator like so:
    data_path = os.path.expanduser(args.path)
    use_cross_validation = args.cv
    constant = args.const
    logreg(data_path, constant, use_cross_validation)

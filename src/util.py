"""
 This is the tool box of processing the data for the three implemented algorithms.

 @Author Feng Long: fxl306@case.edu
 @Date 09/30/2022
 """

import random
import warnings
from typing import Tuple, Iterable
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve



import numpy as np
import math

"""
This is where you will implement helper functions and utility code which you will reuse from project to project.
Feel free to edit the parameters if necessary or if it makes it more convenient.
Make sure you read the instruction clearly to know whether you have to implement a function for a specific assignment.
"""


def count_label_occurrences(y: np.ndarray) -> Tuple[int, int]:
    """
    This is a simple example of a helpful helper method you may decide to implement. Simply takes an array of labels and
    counts the number of positive and negative labels.

    HINT: Maybe a method like this is useful for calculating more complicated things like entropy!

    Args:
        y: Array of binary labels.

    Returns: A tuple containing the number of negative occurrences, and number of positive occurences, respectively.

    """
    n_ones = (y == 1).sum()  # How does this work? What does (y == 1) return?
    n_zeros = y.size - n_ones
    return n_zeros, n_ones


def entropy(y: np.ndarray):
    n_ones, n_zeros = count_label_occurrences(y)
    if n_ones == 0 or n_zeros == 0:
        return 0
    prob0 = n_zeros / len(y)
    prob1 = n_ones / len(y)
    return - prob0 * math.log(prob0, 2) - prob1 * math.log(prob1, 2)


def cv_split(
        X_np: np.ndarray, y: np.ndarray, num_folds: int = 5, stratified: bool = False
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    """
    Conducts a cross-validation split on the given data.

    Args:
        X: Data of shape (n_examples, n_features)
        y: Labels of shape (n_examples,)
        num_folds: Number of CV folds
        stratified:

    Returns: A tuple containing the training data, training labels, testing data, and testing labels, respectively
    for each fold.

    For example, 5 fold cross validation would return the following:
    (
        (X_train_1, y_train_1, X_test_1, y_test_1),
        (X_train_2, y_train_2, X_test_2, y_test_2),
        (X_train_3, y_train_3, X_test_3, y_test_3),
        (X_train_4, y_train_4, X_test_4, y_test_4),
        (X_train_5, y_train_5, X_test_5, y_test_5)
    )

    """

    # Set the RNG seed to 12345 to ensure repeatability
    np.random.seed(12345)
    random.seed(12345)
    rng = np.random.default_rng(seed = 12345)

    X = []

    for x in X_np:
        X.append(x)

    # HINT!
    if stratified:

        # split the data, build the folds
        n_zeros, n_ones = count_label_occurrences(y)
                # size of data of each class in each fold
        fold_class_0_size = int(n_zeros/num_folds)
        fold_class_1_size = int(n_ones/num_folds)
        
        folds = ()
    
        
        for _ in range(0, num_folds-1):
            n_class_0_in_fold = 0
            n_class_1_in_fold = 0
            X_i = None
            y_i = np.array([])
            
            while (n_class_0_in_fold < fold_class_0_size) or (n_class_1_in_fold < fold_class_1_size):
                # choose a random data point
                index = rng.integers(0, len(X))

                if (n_class_0_in_fold < fold_class_0_size and y[index] == 0) or (n_class_1_in_fold < fold_class_1_size and y[index] == 1):
                    if X_i is None:
                        X_i = np.array([X[index]])
                    else:
                        X_i = np.concatenate((X_i, np.array([X[index]])), 0)
                    y_i = np.append(y_i, y[index])
                    if y[index] == 0:
                        n_class_0_in_fold += 1
                    if y[index] == 1:
                        n_class_1_in_fold += 1
                    X.pop(index)
                    y = np.delete(y, index)
            folds += ((X_i,y_i),)
        X = np.array(X)
        folds +=((X, y),)

        # Creating groups by combining folds as training data, and each group uses independent folds as test data
        groups = ()
        for i in range(0, num_folds):
            X_train = None
            y_train = np.array([])
            for j in range(0, num_folds):
                if j != i:
                    if X_train is None:
                        X_train = folds[j][0]
                    else:
                        X_train = np.concatenate((X_train, np.array(folds[j][0])), 0)
                    y_train = np.append(y_train, folds[j][1])
            X_test = folds[i][0].copy()
            y_test = folds[i][1].copy()
            
            groups += ((X_train, y_train, X_test, y_test),)


        return groups

    return (X, y, X, y),


def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Another example of a helper method. Implement the rest yourself!

    Args:
        y: True labels.
        y_hat: Predicted labels.

    Returns: Accuracy
    """

    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')

    n = y.size

    return (y == y_hat).sum() / n


def precision(y: np.ndarray, y_hat: np.ndarray) -> float:
    tp = 0
    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            tp += 1
    return tp / (y_hat == 1).sum()

        


def recall(y: np.ndarray, y_hat: np.ndarray) -> float:
    tp = 0
    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            tp += 1
    return tp / (y == 1).sum()


def roc_curve_pairs(y: np.ndarray, p_y_hat: np.ndarray):
    fpr, tpr, thresholds = roc_curve(y, p_y_hat)
    return fpr, tpr


def auc(y: np.ndarray, p_y_hat: np.ndarray) -> float:
    fpr, tpr = roc_curve_pairs(y, p_y_hat)
    return sklearn.metrics.auc(fpr, tpr)


def printROC(yTrue: np.ndarray, yHat: np.ndarray) -> None:
    fpr, tpr, thersholds = roc_curve(yTrue, yHat, pos_label=1) 

    for i, value in enumerate(thersholds):
        print("%f %f %f" % (fpr[i], tpr[i], value))

    roc_auc = auc(yTrue, yHat)

    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

    plt.xlim([-0.05, 1.05]) 
    plt.ylim([-0.05, 1.05]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title('ROC Curve') 
    plt.legend(loc="lower right") 
    plt.show() 

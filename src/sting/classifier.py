"""top level module for classifier abstractions
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Classifier(ABC):
    """Abstract base class defining common classifier functions.
    Classifier implementations should inherit this class, not instantiate it directly.

    Example:

    ```python
    from sting.classifier import Classifier
    from sting.data import parse_c45
    import numpy as np


    # implements the Classifier base class.
    # must inherit from it and include the fit() and predict() functions
    class MajorityClassifier(Classifier):
        def __init__(self):
            self._class_frequencies: np.ndarray = None

        def fit(self, X: np.ndarray, y: np.ndarray):
            # count the frequencies of each class label
            self._class_frequencies = np.bincount(y)

        def predict(self, X: np.ndarray) -> np.ndarray:
            # predict the most common label for all examples
            return np.full(X.shape[0], self._class_frequencies.argmax())
    ```
        >>> schema, X, y = parse_c45("example")
        >>> model = MajorityClassifier()
        >>> model.fit(X, y)
        >>> y_pred = model.predict(X)
        >>> accuracy = np.sum(y_pred == y) / len(y)
        >>> print(accuracy)
        0.5
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray]) -> None:
        """fit the classifier given a set of examples X with shape (num_examples, num_features) and labels y with shape (num_examples,).

        Args:
            X (np.ndarray): the example set with shape (num_examples, num_features)
            y (np.ndarray): the labels with shape (num_examples,)
            weights (Optional[np.ndarray]): the example weights, if necessary
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """produce a list of output labels for a set of examples X with shape (num_examples, num_features).

        Args:
            X (np.ndarray): examples for which outputs should be provided

        Returns:
            np.ndarray: the predicted outputs with shape (num_examples,)
        """
        pass

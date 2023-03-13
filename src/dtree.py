"""
 This program implements the ID3 decision Tree algorithm that handles nominal and continues attributes,
 and uses both information gain and gain ratio as the split criterion.

 @Author Feng Long: fxl306@case.edu
 @Date 09/30/2022
 """

import argparse
import os.path
import warnings

from typing import Optional, List

import numpy as np
from sting.classifier import Classifier
from sting.data import Feature, parse_c45, FeatureType
import math

import util

nodes_c = 0

# Decision tree for both nominal and continous
class DecisionTree(Classifier):

    # Class store attributes in tree
    class TreeNode:
        pure_node: bool
        children: []
        majority: int
        depth: int
        feature: int
        entropy: int
        members: np.ndarray  # members of label data of current node
        features: List[int]  # not used feature for current node
        cont_features: []  # seperator represented by threshold
        cont_thre: float  # current threshold if node is

        # Constructor for tree node
        def __init__(self, pure_node, children, majority, depth, feature, entropy, members, features, con_features,
                     thre):
            self.depth = depth
            self.majority = majority
            self.pure_node = pure_node
            self.children = children
            self.feature = feature
            self.entropy = entropy
            self.members = members
            self.features = features
            self.cont_features = con_features
            self.cont_thre = thre
            global nodes_c
            nodes_c += 1

    root: TreeNode
    maxDepth: int
    featureSize: int
    act_depth: int

    def __init__(self, schema: List[Feature], depth):
        """
        This is the class where you will implement your decision tree. At the moment, we have provided some dummy code
        where this is simply a majority classifier in order to give you an idea of how the interface works. Don't forget
        to use all the good programming skills you learned in 132 and utilize numpy optimizations wherever possible.
        Good luck!
        """

        # warnings.warn('The DecisionTree class is currently running dummy Majority Classifier code. ' + 'Once you start implementing your decision tree delete this warning message.')

        self._schema = schema  # For some models (like a decision tree) it makes sense to keep track of the data schema
        self._majority_label = 0  # Protected attributes in Python have an underscore prefix
        self.featureSize = len(schema)
        self.maxDepth = depth
        self.act_depth = 0
        global nodes_c
        nodes_c = 0

    # Calculate threshold split point for each continous feature
    def cont_threshold(self, X: np.ndarray, y: np.ndarray, fe: int, member):
        # Connect the value of current feature to current label
        point_of_data = []

        for i in member:
            # lse[0] is feature value, y[i] is label class
            lse = [X[i][fe], y[i]]
            point_of_data.append(lse)

        # Sort base on feature value in ascending order
        point_of_data.sort(key=lambda x: x[0])

        unique_point_of_data = []
        idx = 0

        # Combine all dulicapte data point, if the point has both zero and one as label,
        # marked as -1
        while idx < len(point_of_data):
            lol = point_of_data[idx]
            pre = lol[0]
            flag0 = False
            flag1 = False

            # Combine same data point
            while idx < len(point_of_data) and pre == point_of_data[idx][0]:
                if point_of_data[idx][1] == 1:
                    flag1 = True
                else:
                    flag0 = True
                idx += 1

            # Either one or zero or both
            if flag0 and flag1:
                unique_point_of_data.append([pre, -1])
            elif flag0:
                unique_point_of_data.append([pre, 0])
            else:
                unique_point_of_data.append([pre, 1])


        # The starting value and label
        pre = unique_point_of_data[0][1]
        pre_v = unique_point_of_data[0][0]
        threshold_set: set[float] = set()

        # Loop through data points to find split threshold (0 -> 1 or 1 -> 0)
        for index in range(1, len(unique_point_of_data)):
            ls = unique_point_of_data[index]

            if pre != ls[1] or pre == -1 or ls[1] == -1:
                threshold_set.add((ls[0] + pre_v) / 2.0)

            pre = ls[1]
            pre_v = ls[0]

        thresholds = []

        for thre in threshold_set:
            thresholds.append(thre)

        thresholds.sort()
        return thresholds

    # Train the decision tree
    def fit(self, X: np.ndarray, y: np.ndarray, use_GR = False, weights: Optional[np.ndarray] = None) -> None:
        """
        This is the method where the training algorithm will run.

        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            weights: Weights for each example. Will become relevant later in the course, ignore for now.
        """

        try:
            # Add all labels and features to current node
            features = []
            members = []

            for x in range(len(self.schema)):
                features.append(x)

            for x in range(len(y)):
                members.append(x)

            # Entropy of set of labels without and classification
            root_entropy = util.entropy(y)

            # The threshold for each continues feature of current members
            continues_feature_threshold = []

            # calculate the threshold for each continues feature of current members
            for feature_index in range(len(self.schema)):
                cur_feature = self.schema[feature_index]

                # Only calculate threshold if feature is CONTINUOUS
                if cur_feature.ftype == FeatureType.CONTINUOUS:
                    lists = self.cont_threshold(X, y, feature_index, members)
                    continues_feature_threshold.append(lists)
                else:
                    continues_feature_threshold.append([])

            # Create the root of the decision tree
            self.root = self.TreeNode(False, [], -1, 0, 0, root_entropy, members, features, continues_feature_threshold, -1)
            # Start split from the root
            self._determine_split_criterion_and_pick_feature(X, y, self.root, use_GR)
        except NotImplementedError:
            warnings.warn("Missing functions")

    # Count zeros and ones for the feature
    def count_label_occurrences_member(self, member, label):
        one = 0
        zero = 0

        # loop all data to find ones and zeros
        for x in member:
            if label[x] == 1:
                one += 1
            else:
                zero += 1

        return one, zero

    # Find label of each member
    def label_of_members(self, members, y):
        new_y = []
        for i in members:
            new_y.append(y[i])
        return np.array(new_y)

    # IG for nomial feature
    def info_gain(self, label: np.ndarray, attributes, parent_entropy, member_nums):
        new_entropy = []
        child_entropy = 0

        # Gain Ratio
        HX = 0

        # Calculate entropy for each attribute of current feature
        for arr in attributes:
            if len(arr) == 0:
                new_entropy.append(0)
            else:
                # Calculate one and zeros to find entropy
                prob_of_value = len(arr) / member_nums
                labels_for_each_value = self.label_of_members(arr, label)
                new_entropy.append(util.entropy(labels_for_each_value))
                child_entropy += new_entropy[-1] * prob_of_value

                # Cal gain ratio by adding entropy of each member set
                HX += (-1 * prob_of_value * math.log2(prob_of_value))

        return parent_entropy - child_entropy, new_entropy, HX

    # Calculte majority label of current node
    def cal_majority(self, y: np.ndarray, members: np.ndarray):
        one = 0
        zero = 0

        for member in members:
            if y[member] == 0:
                zero += 1
            else:
                one += 1

        return zero, one

    # Calcuate member of each attribute of nominal and binary feature
    def cal_member(self, feature_index, feature: Feature, X: np.ndarray, member, y):

        if feature.ftype == FeatureType.BINARY:
            new_member = [[], []]

            for x in member:
                if X[x][feature_index] == 0:
                    new_member[0].append(x)
                else:
                    new_member[1].append(x)

            return new_member

        elif feature.ftype == FeatureType.NOMINAL:
            new_member = []

            # Append number of attributes to the current list
            for x in range(len(feature.values)):
                new_member.append([])

            # Add corresponding member to each attribute
            for cc in member:
                if not np.isnan(cc):
                    ind = int(X[cc][feature_index])
                    new_member[ind - 1].append(cc)

            return new_member

    # Find member after split for continuos
    def split_member(self, thre, X: np.ndarray, feature_index, memeber):
        before = []
        after = []

        for m in memeber:
            if X[m][feature_index] < thre:
                before.append(m)
            else:
                after.append(m)

        return before, after

    def cal_num_ones_zeros_before_thresholds(self, X: np.ndarray, y: np.ndarray, memeber, feature_index, thres):
        lists = []

        for m in memeber:
            lists.append([X[m][feature_index], y[m]])

        lists.sort(key=lambda x: x[0])

        res = []
        ones = 0
        zeros = 0
        index = 0

        for the in thres:
            while index < len(lists) and lists[index][0] <= the:
                if lists[index][1] == 0:
                    zeros += 1
                else:
                    ones += 1

                index += 1

            res.append([ones, zeros])

        return res

    # Calculate total member that have label one and label zero for one feature
    def cal_totals_ones_zeros(self, member, y):
        ones = 0
        zeros = 0

        for m in member:
            if y[m] == 1:
                ones += 1
            else:
                zeros += 1

        return ones, zeros

    # Calculate number of ones and zeros before each split threshold
    def cal_one_zeros_before(self, X: np.ndarray, y: np.ndarray, memeber, feature_index, thresholds):
        lists = []

        # Find corresponding feature value of each data
        for m in memeber:
            lists.append([X[m][feature_index], y[m]])

        # Sort based on feature value in ascending order
        lists.sort(key=lambda x: x[0])

        res = []
        ones = 0
        zeros = 0
        index = 0

        # Find ones and zeros by looping through thresholds
        for threshold in thresholds:
            while index < len(lists) and lists[index][0] <= threshold:
                if lists[index][1] == 0:
                    zeros += 1
                else:
                    ones += 1

                index += 1

            # push ones and zeros for each threshold
            res.append([ones, zeros])

        return res

    # Calculate entropy of each threshold in split
    def cal_entropy_cont(self, n_ones, n_zeros):
        size = n_ones + n_zeros

        # no member in current interval
        if size == 0:
            return 0

        prob0 = n_zeros / size
        prob1 = n_ones / size
        entropy = 0

        # This means the node is pure
        if prob0 == 0 or prob1 == 0:
            return 0
        else:
            entropy += (-1 * prob0 * math.log2(prob0))
            entropy += (-1 * prob1 * math.log2(prob1))

        return entropy

    # For the member in current feature, find which members are smaller/greater than threshold
    def cal_member_cont(self, thre, X: np.ndarray, feature_index, memeber):
        before = []
        after = []

        # For the member in current, find which members are smaller/greater than threshold
        for m in memeber:
            if X[m][feature_index] <= thre:
                before.append(m)
            else:
                after.append(m)

        return before, after

    # IG for continous
    def info_gain_cont(self, node: TreeNode, X: np.ndarray, y: np.ndarray, feature_index, member):
        # All threshold separation of current feature
        threshold_separation = node.cont_features[feature_index]
        # New entropy after split using the best threshold separation
        new_entropy = [0, 0]
        # Max information gain
        max_ig = -1
        best_threshold_index = -1
        # Prob of distribution
        prob1_dis = 0
        prob2_dis = 0

        # Calculate total member that have label one and label zero
        t_ones, t_zeros = self.cal_totals_ones_zeros(member, y)
        # Calculate number of ones and zeros before each split threshold
        ones_zeros_before_thresholds = self.cal_one_zeros_before(X, y, node.members, feature_index, threshold_separation)

        # For each threshold split point, calculate corresponding entropy and information gain
        for index in range(len(ones_zeros_before_thresholds)):
            # Ones and zeros before current threshold
            ones_zeros = ones_zeros_before_thresholds[index]

            # Prob of members before and after current threshold
            prob1 = (ones_zeros[0] + ones_zeros[1]) / len(member)
            prob2 = (len(member) - ones_zeros[0] - ones_zeros[1]) / len(member)

            # Entropy of members before and after current threshold
            enr1 = self.cal_entropy_cont(ones_zeros[0], ones_zeros[1])
            enr2 = self.cal_entropy_cont(t_ones - ones_zeros[0], t_zeros - ones_zeros[1])

            # H(Y|X) = P(X=1)H(X1) + P(X=0)H(X0)
            current_entropy = prob1 * enr1 + prob2 * enr2

            # Check if current IG is the best IG
            if node.entropy - current_entropy > max_ig:
                new_entropy[0] = enr1
                new_entropy[1] = enr2
                max_ig = node.entropy - current_entropy
                best_threshold_index = index
                prob1_dis = prob1
                prob2_dis = prob2

        # Calculate member before threshold and after threshold
        if best_threshold_index >= 0:
            before_threshold, after_threshold = \
                self.cal_member_cont(threshold_separation[best_threshold_index], X, feature_index, member)
        else:
            before_threshold, after_threshold = [], []

        # Cal gain ratio
        if prob1_dis == 0 or prob2_dis == 0:
            HX = 0
        else:
            HX = (-1 * prob1_dis * math.log2(prob1_dis)) + (-1 * prob2_dis * math.log2(prob2_dis))

        return max_ig, [before_threshold, after_threshold], new_entropy, best_threshold_index, HX

    # Pick the feature with max ig each time
    def _determine_split_criterion_and_pick_feature(self, X: np.ndarray, y: np.ndarray, parent: TreeNode, use_GR: False):
        # conditions that turns current node as pure node
        if parent.depth == self.maxDepth or len(parent.features) == 0 or parent.pure_node \
                or parent.entropy == 0 or len(parent.members) == 0:
            parent.pure_node = True
            zero, one = self.cal_majority(y, parent.members)

            if zero >= one:
                parent.majority = 0
            else:
                parent.majority = 1

            return

        largest_ig = -100
        best_feature = -1
        next_member: []
        next_entro: []
        # If feature is continous, also find the best split point
        best_split_threshold_index = -1
        next_continues_threshold = []

        # calculate the threshold for each continues feature of current members
        for feature_index in range(len(self.schema)):
            cur_feature = self.schema[feature_index]

            # Only calculate threshold if feature is CONTINUOUS
            if cur_feature.ftype == FeatureType.CONTINUOUS:
                lists = self.cont_threshold(X, y, feature_index, parent.members)
                next_continues_threshold.append(lists)
            else:
                next_continues_threshold.append([])

        # The split threshold point of each continous feature
        parent.cont_features = next_continues_threshold

        # Iterate all features that has not been picked so far to find the next best feature
        for f in parent.features:
            if self.schema[f].ftype == FeatureType.CONTINUOUS:
                cur_ig, new_member, cur_entro, current_split_threshold_index, HX = self.info_gain_cont(parent, X, y, f, parent.members)

            else:
                if (self.schema[f].name == 'image_id'):
                    # exclude id feature, Comment out if using gain ratio
                    continue

                new_member = self.cal_member(f, self._schema[f], X, parent.members, y)
                cur_ig, cur_entro, HX = self.info_gain(y, new_member, parent.entropy, len(parent.members))
                current_split_threshold_index = -1

            # If use gain ratio, calculate
            if use_GR and HX != 0:
                cur_ig = cur_ig / HX

            if cur_ig >= largest_ig:
                next_entro = cur_entro
                next_member = new_member
                largest_ig = cur_ig
                best_feature = f
                best_split_threshold_index = current_split_threshold_index

        # This should not happen, just in case all features increase entropy
        if best_feature == -1:
            parent.pure_node = True
            zero, one = self.cal_majority(y, parent.members)

            if zero >= one:
                parent.majority = 0
            else:
                parent.majority = 1
            return

        parent.feature = best_feature
        next_features = parent.features.copy()

        if self.schema[best_feature].ftype == FeatureType.CONTINUOUS:
            # Change threshold to the best threshold
            next_cont_threshold = parent.cont_features[best_feature][best_split_threshold_index]
            parent.cont_thre = next_cont_threshold
        else:
            next_features.remove(best_feature)

        # Add children from attritube of features
        for x in range(len(next_entro)):
            # Current entropy is equal to zero, pure node
            if next_entro[x] == 0:
                node = self.TreeNode(True, [], -1, parent.depth + 1, -1, next_entro[x], next_member[x],
                                     next_features, [], -1)
            else:
                node = self.TreeNode(False, [], -1, parent.depth + 1, -1, next_entro[x], next_member[x],
                                     next_features, [], -1)

            # Add next tree node as child of children
            parent.children.append(node)

            # Recursive calling on remaining features
            self._determine_split_criterion_and_pick_feature(X, y, node, use_GR)

    # Run test data on dtree
    def prediction(self, x: np.ndarray, node: TreeNode) -> int:
        self.act_depth = max(self.act_depth, node.depth)

        # If pure node return the majority label of this node asa result
        if node.pure_node:
            return node.majority

        ind = node.feature
        n = int(x[ind])

        # Compare current feature value with threshold if the feature if CONTINUOUS
        if self.schema[node.feature].ftype == FeatureType.CONTINUOUS:
            if x[ind] < node.cont_thre:
                n = 1
            else:
                n = 2

        # Find next child and recurse
        next_child = node.children[n - 1]
        return self.prediction(x, next_child)

    # Run dtree
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This is the method where the decision tree is evaluated.

        Args:
            X: The testing data of shape (n_examples, n_features).

        Returns: Predictions of shape (n_examples,), either 0 or 1
        """

        result = []

        # Run test data
        for x in range(len(X)):
            result.append(self.prediction(X[x], self.root))

        return np.array(result)

        # Returns either all 1s or all 0s, depending on _majority_label.
        # return np.ones(X.shape[0], dtype=np.int) * self._majority_label

    # In Python, instead of getters and setters we have properties: docs.python.org/3/library/functions.html#property
    @property
    def schema(self):
        """
        Returns: The dataset schema
        """
        return self._schema


def evaluate_and_print_metrics(dtree: DecisionTree, X: np.ndarray, y: np.ndarray):
    """
    You will implement this method.
    Given a trained decision tree and labelled dataset, Evaluate the tree and print metrics.
    """

    y_hat = dtree.predict(X)
    acc = util.accuracy(y, y_hat)
    print(f'Accuracy:{acc:.2f}')
    print('Size:', nodes_c)
    print('Maximum Depth:', dtree.act_depth)
    print('First Feature:', dtree.schema[dtree.root.feature].name)


def dtree(data_path: str, tree_depth_limit: int, use_cross_validation: bool = True, information_gain: bool = True):
    """
    It is highly recommended that you make a function like this to run your program so that you are able to run it
    easily from a Jupyter notebook. This function has been PARTIALLY implemented for you, but not completely!

    :param data_path: The path to the data.
    :param tree_depth_limit: Depth limit of the decision tree
    :param use_cross_validation: If True, use cross validation. Otherwise, run on the full dataset.
    :param information_gain: If true, use information gain as the split criterion. Otherwise use gain ratio.
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

    use_GR = not information_gain
    for X_train, y_train, X_test, y_test in datasets:
        decision_tree = DecisionTree(schema, tree_depth_limit)
        decision_tree.fit(X_train, y_train, use_GR)
        evaluate_and_print_metrics(decision_tree, X_test, y_test)


if __name__ == '__main__':
    """
    THIS IS YOUR MAIN FUNCTION. You will implement the evaluation of the program here. We have provided argparse code
    for you for this assignment, but in the future you may be responsible for doing this yourself.
    """

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a decision tree algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('depth_limit', metavar='DEPTH', type=int,
                        help='Depth limit of the tree. Must be a non-negative integer. A value of 0 sets no limit.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.add_argument('--use-gain-ratio', dest='gain_ratio', action='store_true',
                        help='Use gain ratio as tree split criterion instead of information gain.')
    parser.set_defaults(cv=True, gain_ratio=False)
    args = parser.parse_args()

    # If the depth limit is negative throw an exception
    if args.depth_limit < 0:
        raise argparse.ArgumentTypeError('Tree depth limit must be non-negative.')

    # You can access args with the dot operator like so:
    data_path = os.path.expanduser(args.path)
    tree_depth_limit = args.depth_limit
    use_cross_validation = args.cv
    use_information_gain = not args.gain_ratio

    if tree_depth_limit == 0:
        tree_depth_limit = 1000

    dtree(data_path, tree_depth_limit, use_cross_validation, use_information_gain)

import numpy as np


class Node:

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        gain=None,
        value=None
    ):

        self.feature = feature
        self.threshold = threshold

        self.left = left
        self.right = right

        self.gain = gain

        self.value = value


class DecisionTreeRegressor:

    def __init__(
        self,
        min_samples_split=2,
        max_depth=5
    ):

        self.root = None

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def _split(self, feature_values, threshold):

        left_idx = np.where(feature_values <= threshold)[0]
        right_idx = np.where(feature_values > threshold)[0]

        return left_idx, right_idx

    def _leaf_value(self, y):

        return np.mean(y)

    def _variance(self, y):

        if len(y) == 0:
            return 0

        mean = np.mean(y)

        variance = np.mean((y - mean) ** 2)

        return variance

    def _variance_reduction(
        self,
        y,
        feature_values,
        threshold
    ):

        parent_variance = self._variance(y)

        left_idx, right_idx = self._split(
            feature_values,
            threshold
        )

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        n = len(y)

        n_left = len(left_idx)
        n_right = len(right_idx)

        left_variance = self._variance(
            y[left_idx]
        )

        right_variance = self._variance(
            y[right_idx]
        )

        weighted_variance = (

            (n_left / n) * left_variance

            +

            (n_right / n) * right_variance

        )

        variance_reduction = (

            parent_variance

            -

            weighted_variance

        )

        return variance_reduction

    def _best_split(self, X, y, n_features):

        best_split = {}
        best_gain = -1

        for feature in range(n_features):

            feature_values = X[:, feature]

            thresholds = np.unique(feature_values)

            for threshold in thresholds:

                gain = self._variance_reduction(
                    y,
                    feature_values,
                    threshold
                )

                if gain > best_gain:

                    left_idx, right_idx = self._split(
                        feature_values,
                        threshold
                    )

                    best_gain = gain

                    best_split = {

                        "feature": feature,

                        "threshold": threshold,

                        "gain": gain,

                        "left_dataset": {

                            "X": X[left_idx],

                            "y": y[left_idx]

                        },

                        "right_dataset": {

                            "X": X[right_idx],

                            "y": y[right_idx]

                        }

                    }

        return best_split

    def _build_tree(self, X, y, depth=0):

        n_samples = X.shape[0]
        n_features = X.shape[1]

        if (

            n_samples < self.min_samples_split

            or

            depth >= self.max_depth

        ):

            leaf_value = self._leaf_value(y)

            return Node(value=leaf_value)

        best_split = self._best_split(
            X,
            y,
            n_features
        )

        if not best_split or best_split["gain"] <= 0:

            leaf_value = self._leaf_value(y)

            return Node(value=leaf_value)

        left_subtree = self._build_tree(

            best_split["left_dataset"]["X"],

            best_split["left_dataset"]["y"],

            depth + 1

        )

        right_subtree = self._build_tree(

            best_split["right_dataset"]["X"],

            best_split["right_dataset"]["y"],

            depth + 1

        )

        return Node(

            feature=best_split["feature"],

            threshold=best_split["threshold"],

            left=left_subtree,

            right=right_subtree,

            gain=best_split["gain"]

        )

    def fit(self, X, y):

        if len(X) != len(y):

            raise ValueError(
                "Number of samples in X and y must be equal."
            )

        X = np.array(X)
        y = np.array(y)

        self.root = self._build_tree(
            X,
            y
        )

    def _predict(self, x):

        node = self.root

        while node.value is None:

            if x[node.feature] <= node.threshold:

                node = node.left

            else:

                node = node.right

        return node.value

    def predict(self, X):

        X = np.array(X)

        if X.ndim == 1:

            X = X.reshape(1, -1)

        predictions = []

        for x in X:

            prediction = self._predict(x)

            predictions.append(prediction)

        return np.array(predictions)
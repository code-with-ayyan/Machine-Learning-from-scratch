import numpy as np 
from collections import Counter


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
        
    

class CustomDecisionTreeClassifier:

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
    
    

    def _most_common_label(self, y):

        counter = Counter(y)

        value = counter.most_common(1)[0][0]

        return value
    
    def _entropy(self, y):

        classes = np.unique(y)

        entropy = 0

        for c in classes:

            p = len(y[y==c])/len(y)

            if p > 0:
                entropy -= p*np.log2(p)

        return entropy
    
    
    def _information_gain(
    self,
    y,
    feature_values,
    threshold
):

        parent_entropy = self._entropy(y)

        left_idx, right_idx = self._split(
            feature_values,
            threshold
        )

        if len(left_idx) == 0 or len(right_idx) == 0:

            return 0

        n = len(y)

        n_left = len(left_idx)
        n_right = len(right_idx)

        left_entropy = self._entropy(
            y[left_idx]
        )

        right_entropy = self._entropy(
            y[right_idx]
        )

        child_entropy = (

            (n_left / n) * left_entropy

            +

            (n_right / n) * right_entropy

        )

        information_gain = (

            parent_entropy

            -

            child_entropy

        )

        return information_gain
        
        
    def _best_split(self, X, y, n_features):

        best_split = {}

        best_gain = -1

        for feature in range(n_features):

            feature_values = X[:, feature]

            thresholds = np.unique(feature_values)

            for threshold in thresholds:

                gain = self._information_gain(
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
        n_classes = len(np.unique(y))

        
        if (
            n_samples < self.min_samples_split
            or depth >= self.max_depth
            or n_classes == 1
        ):

            leaf_value = self._most_common_label(y)

            return Node(value=leaf_value)

        
        best_split = self._best_split(X, y, n_features)

        
        if not best_split or best_split["gain"] <= 0:

            leaf_value = self._most_common_label(y)

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
        
        if len(X)!=len(y):

            raise ValueError(
                "Number of samples in X and y must be equal."
            )

        X = np.array(X)
        y = np.array(y)
        
        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

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
        
        if self.root is None:
            raise ValueError(
                "Model is not fitted yet."
            )

        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        

        predictions = []

        for x in X:

            prediction = self._predict(x)

            predictions.append(prediction)

        return np.array(predictions)
    
    
    
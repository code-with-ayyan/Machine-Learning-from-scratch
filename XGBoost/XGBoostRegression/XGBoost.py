import numpy as np


class GBTreeNode:

    def __init__(
        self,
        value=None,
        feature=None,
        threshold=None,
        left=None,
        right=None
    ):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None


class GBTree:

    def __init__(
        self,
        max_depth=3,
        max_leaf_nodes=None,
        min_child_weight=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0
    ):

        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be >= 0")

        if max_leaf_nodes is not None and max_leaf_nodes < 1:
            raise ValueError("max_leaf_nodes must be >= 1")

        if min_child_weight < 0:
            raise ValueError("min_child_weight must be >= 0")

        if reg_lambda < 0:
            raise ValueError("reg_lambda must be >= 0")

        if reg_alpha < 0:
            raise ValueError("reg_alpha must be >= 0")

        if gamma < 0:
            raise ValueError("gamma must be >= 0")

        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma

        self.root = None
        self.n_leaves = 0


    def _calculate_leaf_value(
        self,
        gradients,
        hessians
    ):

        G = gradients.sum()
        H = hessians.sum()

        if self.reg_alpha != 0:

            if G > self.reg_alpha:
                G = G - self.reg_alpha

            elif G < -self.reg_alpha:
                G = G + self.reg_alpha

            else:
                G = 0.0

        value = -G / (
            H + self.reg_lambda
        )

        return value


    def _node_score(
        self,
        gradients,
        hessians
    ):

        G = gradients.sum()
        H = hessians.sum()

        if self.reg_alpha != 0:

            if G > self.reg_alpha:
                G = G - self.reg_alpha

            elif G < -self.reg_alpha:
                G = G + self.reg_alpha

            else:
                G = 0.0

        return (
            G ** 2
        ) / (
            H + self.reg_lambda
        )


    def _calculate_gain(
        self,
        gradients,
        hessians,
        left_mask,
        right_mask
    ):

        left_gradients = gradients[left_mask]
        left_hessians = hessians[left_mask]

        right_gradients = gradients[right_mask]
        right_hessians = hessians[right_mask]

      
        if (
            left_hessians.sum()
            < self.min_child_weight
        ):
            return -np.inf

        if (
            right_hessians.sum()
            < self.min_child_weight
        ):
            return -np.inf

        parent_score = self._node_score(
            gradients,
            hessians
        )

        left_score = self._node_score(
            left_gradients,
            left_hessians
        )

        right_score = self._node_score(
            right_gradients,
            right_hessians
        )

        gain = 0.5 * (
            left_score
            + right_score
            - parent_score
        )

     
        gain -= self.gamma

        return gain


    def _find_best_split(
        self,
        X,
        gradients,
        hessians
    ):

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature in range(n_features):

            values = np.unique(
                X[:, feature]
            )

            if len(values) < 2:
                continue

            # Midpoints between unique values
            thresholds = (
                values[:-1]
                + values[1:]
            ) / 2

            for threshold in thresholds:

                left_mask = (
                    X[:, feature] <= threshold
                )

                right_mask = (
                    X[:, feature] > threshold
                )

                gain = self._calculate_gain(
                    gradients,
                    hessians,
                    left_mask,
                    right_mask
                )

                if gain > best_gain:

                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return (
            best_feature,
            best_threshold,
            best_gain
        )


    def _build_tree(
        self,
        X,
        gradients,
        hessians,
        depth=0
    ):
        
        leaf_value = self._calculate_leaf_value(
            gradients,
            hessians
        )



        if (
            self.max_depth is not None
            and depth >= self.max_depth
        ):

            self.n_leaves += 1

            return GBTreeNode(
                value=leaf_value
            )



        if (
            self.max_leaf_nodes is not None
            and self.n_leaves + 2
            > self.max_leaf_nodes
        ):

            self.n_leaves += 1

            return GBTreeNode(
                value=leaf_value
            )

        (
            feature,
            threshold,
            gain
        ) = self._find_best_split(
            X,
            gradients,
            hessians
        )



        if (
            feature is None
            or gain <= 0
        ):

            self.n_leaves += 1

            return GBTreeNode(
                value=leaf_value
            )



        left_mask = (
            X[:, feature] <= threshold
        )

        right_mask = (
            X[:, feature] > threshold
        )


        if (
            left_mask.sum() == 0
            or right_mask.sum() == 0
        ):

            self.n_leaves += 1

            return GBTreeNode(
                value=leaf_value
            )



        left_child = self._build_tree(
            X[left_mask],
            gradients[left_mask],
            hessians[left_mask],
            depth + 1
        )



        right_child = self._build_tree(
            X[right_mask],
            gradients[right_mask],
            hessians[right_mask],
            depth + 1
        )


       

        return GBTreeNode(
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child
        )


    def fit(
        self,
        X,
        gradients,
        hessians
    ):

        X = np.asarray(X)
        gradients = np.asarray(
            gradients,
            dtype=float
        )
        hessians = np.asarray(
            hessians,
            dtype=float
        )


        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array"
            )

        if gradients.ndim != 1:
            raise ValueError(
                "gradients must be a 1D array"
            )

        if hessians.ndim != 1:
            raise ValueError(
                "hessians must be a 1D array"
            )

        if len(X) == 0:
            raise ValueError(
                "X cannot be empty"
            )

        if (
            len(X)
            != len(gradients)
            or
            len(X)
            != len(hessians)
        ):
            raise ValueError(
                "X, gradients and hessians "
                "must have the same number of samples"
            )


        self.root = None
        self.n_leaves = 0


        self.root = self._build_tree(
            X,
            gradients,
            hessians
        )

        return self


    def _predict(
        self,
        x,
        node
    ):

        if node.is_leaf():
            return node.value


        if x[node.feature] <= node.threshold:

            return self._predict(
                x,
                node.left
            )


        return self._predict(
            x,
            node.right
        )


    def predict(self, X):

        if self.root is None:
            raise ValueError(
                "GBTree has not been fitted yet"
            )

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array"
            )

        return np.array([
            self._predict(
                x,
                self.root
            )
            for x in X
        ])


# ===========================
# XGBoost Regressor
# ===========================


class XGBoostRegressorCustom:

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        max_leaf_nodes=None,
        min_child_weight=1,
        gamma=0,
        reg_lambda=1,
        reg_alpha=0
    ):

        if n_estimators < 1:
            raise ValueError(
                "n_estimators must be >= 1"
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be > 0"
            )

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes

        self.min_child_weight = min_child_weight

        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha

        self.trees = []
        self.base_prediction = None


    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(
            y,
            dtype=float
        )


        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array"
            )

        if y.ndim != 1:
            raise ValueError(
                "y must be a 1D array"
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must have "
                "the same number of samples"
            )

        if len(y) == 0:
            raise ValueError(
                "X and y cannot be empty"
            )

        self.trees = []



        self.base_prediction = y.mean()

        prediction = np.full(
            len(y),
            self.base_prediction,
            dtype=float
        )



        for _ in range(
            self.n_estimators
        ):


            gradients = (
                prediction - y
            )

            hessians = np.ones_like(
                y,
                dtype=float
            )



            tree = GBTree(
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha
            )


            tree.fit(
                X,
                gradients,
                hessians
            )


            tree_update = tree.predict(X)

            prediction += (
                self.learning_rate
                * tree_update
            )


            self.trees.append(tree)


        return self


    def _predict_raw(self, X):

        if self.base_prediction is None:
            raise ValueError(
                "XGBoostRegressorCustom "
                "has not been fitted yet"
            )


        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array"
            )


        prediction = np.full(
            len(X),
            self.base_prediction,
            dtype=float
        )


        for tree in self.trees:

            tree_update = tree.predict(X)

            prediction += (
                self.learning_rate
                * tree_update
            )


        return prediction


    def predict(self, X):

        return self._predict_raw(X)
              
        
        
        
        
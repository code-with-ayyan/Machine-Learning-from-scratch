import numpy as np 
from sklearn.tree import DecisionTreeRegressor 


class CustomGradientBoostingCLassifier:
    
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 random_state=None,
                 max_leaf_node=3,
                 max_depth=None
                ):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_node
        self.random_state = random_state

        self.initial_log_odds = None
        self.estimators = []
        self.leaf_values = []
        
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
        
    def fit(self, X, y):
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        positive_ratio = np.mean(y)
        
        self.initial_log_odds = np.log(
            positive_ratio / (1 - positive_ratio)
        )
        
        log_odds = np.full(
            len(y),
            self.initial_log_odds,
            dtype=float
        )

        
        for i in range(self.n_estimators):

            probabilities = self._sigmoid(log_odds)

            residuals = y - probabilities
            
            hessian = probabilities * (1 - probabilities)

            tree = DecisionTreeRegressor(
                max_leaf_nodes=self.max_leaf_nodes,
                max_depth =self.max_depth,
                random_state=self.random_state
            )

            tree.fit(X, residuals)
            
            leaf_indices = tree.apply(X)

            current_leaf_values = {}
            
            for leaf in np.unique(leaf_indices):

                mask = leaf_indices == leaf

                numerator = np.sum(
                    residuals[mask]
                )
                

                denominator = np.sum(
                    hessian[mask]
                )

                if denominator == 0:
                    gamma = 0.0
                else:
                    gamma = numerator / denominator

                current_leaf_values[leaf] = gamma
            
            tree_update = np.array([
                current_leaf_values[leaf]
                for leaf in leaf_indices
            ])

            log_odds += (
                self.learning_rate *
                tree_update
            )

            self.estimators.append(tree)
            self.leaf_values.append(
                current_leaf_values
            )

        return self
            
    def predict_proba(self, X):

        X = np.asarray(X)

        log_odds = np.full(
            len(X),
            self.initial_log_odds,
            dtype=float
        )

        for tree, leaf_values in zip(
            self.estimators,
            self.leaf_values
        ):

            leaves = tree.apply(X)

            tree_update = np.array([
                leaf_values.get(leaf, 0.0)
                for leaf in leaves
            ])

            log_odds += (
                self.learning_rate *
                tree_update
            )


        probabilities = self._sigmoid(log_odds)

        return np.column_stack([
            1 - probabilities,
            probabilities
        ])  
        
    
    def predict(self, X):

        probabilities = self.predict_proba(X)[:, 1]

        return (probabilities >= 0.5).astype(int)
            
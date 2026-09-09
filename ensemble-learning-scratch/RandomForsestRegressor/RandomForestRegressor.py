import numpy as np 
from DecisionTreeRegressor import CustomDecisionTreeRegressor




class CustomRandomForestRegressor:
    
    def __init__(self,
                n_estimators = 10,
                min_samples_split=2,
                max_depth=5,
                bootstrap= True,
                max_samples= 0.6,
                max_features='sqrt'):
        
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.max_samples = max_samples
        
        self.estimators_ = []
        
        
    def bootstrap_sample(self, X, y):
    
            X = np.asarray(X)
            y = np.asarray(y)
    
            n_samples = X.shape[0]
    
            indices = np.random.choice(
                n_samples,
                size=int(self.max_samples * n_samples),
                replace=self.bootstrap
            )
    
            return X[indices], y[indices]
        
        
        
    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if len(X) != len(y):
            raise ValueError(
                "Number of samples in X and y must be equal."
            )

        self.estimators_ = []

        for _ in range(self.n_estimators):

            X_sample, y_sample = self.bootstrap_sample(
                X, y
            )

            tree = CustomDecisionTreeRegressor(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                max_features=self.max_features
            )

            tree.fit(
                X_sample,
                y_sample
            )

            self.estimators_.append(tree)

        return self
    
    def _predict(self,X):
        
        predictions = []
        
        for tree in self.estimators_:
            
            prediction = tree.predict(X)
            predictions.append(prediction)
            
        return np.array(predictions)
    
    def predict(self, X):

        if len(self.estimators_) == 0:
            raise ValueError(
                "Model is not fitted yet."
            )

        predictions = self._predict(X)

        mean = np.mean(predictions, axis = 0)

        return mean
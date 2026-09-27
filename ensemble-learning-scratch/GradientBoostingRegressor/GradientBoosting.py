import numpy as np 
from copy import deepcopy

class CustomGradientBoostingRegressor:
    
    def __init__(self,base_estimator, n_estimators, learning_rate = 0.1):
        
        self.base_estimator = base_estimator
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        
        
        self.initial_prediction = None
        self.estimators = []
        
        self.residuals = []
        
    def fit(self, X, y):
        
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        
        
        self.initial_prediction = y.mean()
        
        prediction = np.full(len(y), y.mean())
        
        for _ in range(self.n_estimators):
            
            residual = y - prediction
            
            self.residuals.append(residual)
            
            model = deepcopy(self.base_estimator)
            
            model.fit(X, residual)
            
            model_pred = model.predict(X)
            
            prediction += (self.learning_rate * model_pred)
            
            self.estimators.append(model)
            
            
        return self  
    
    def residuals_(self):
        
        return np.array(self.residuals)
        
    def predict(self, X):
        
        X = np.asarray(X)
        
        predictions = np.full(len(X), self.initial_prediction)
        
        for estimator in self.estimators:
            
            predictions += (self.learning_rate * estimator.predict(X))
            
        return predictions
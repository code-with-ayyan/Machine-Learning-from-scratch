import numpy as np 


class KNN_Regressor():

    def __init__(self,k = 3):
        self.k = k 
        
    def fit(self,X,y):
        
        if self.k > len(X):
            raise ValueError(
            "k cannot be greater than the number of training samples."
        )
        
        if len(X) != len(y):
            raise ValueError(
            "Number of samples in X and y must be equal."
        )
        
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def euclidean_distance(self,x1,x2):
        distance = np.sqrt(np.sum((x1 - x2) ** 2))
        return distance 
    
    def _predict(self,x):
        
        distances = []
        
        for x_train in self.X_train:
            distance = self.euclidean_distance(x_train,x)
            distances.append(distance)
        
        distances = np.array(distances)
        
        k_indices = np.argsort(distances)[:self.k]
        
        k_values = [self.y_train[i] for i in k_indices]
        
        prediction = np.mean(k_values)
        
        return prediction 
    
    def predict(self, X):
        
        X = np.array(X)
        predictions = []
        
        for x in X:
            prediction = self._predict(x)
            predictions.append(prediction)
            
        return np.array(predictions)
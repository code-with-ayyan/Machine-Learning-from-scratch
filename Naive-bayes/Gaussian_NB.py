import numpy as np 

class GaussianNB():
    
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
    
    def fit(self,X,y):
        
        if len(X) != len(y):
            raise ValueError(
                "Number of samples in X and y must be equal."
            )
            
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):

            X_c = X[y == c]

            self.mean[idx] = np.mean(X_c, axis=0)

            self.var[idx] = np.var(X_c, axis=0)

            self.priors[idx] = len(X_c) / len(X)
            
    def _gaussian_pdf(self, class_idx, x):

        mean = self.mean[class_idx]
        var = self.var[class_idx]

        epsilon = 1e-9
        var += epsilon

        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return np.maximum(numerator / denominator, 1e-9)
    
    def _predict(self, x):

        posteriors = []

        for idx, c in enumerate(self.classes):

            prior = np.log(self.priors[idx])

            class_conditional = np.sum(
                np.log(self._gaussian_pdf(idx, x))
            )

            posterior = prior + class_conditional

            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        
        if self.mean is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.array(X)

        predictions = []

        for x in X:
            prediction = self._predict(x)
            predictions.append(prediction)

        return np.array(predictions)
    
import numpy as np
from copy import deepcopy


class CustomBaggingRegressor:

    def __init__(
        self,
        base_estimator,
        n_estimators=10,
        max_samples=1.0,
        bootstrap=True
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
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

        self.estimators_ = []

        for _ in range(self.n_estimators):

            X_sample, y_sample = self.bootstrap_sample(X, y)

            estimator = deepcopy(self.base_estimator)

            estimator.fit(X_sample, y_sample)

            self.estimators_.append(estimator)

        return self

    def _predict(self, X):

        predictions = np.array([
            estimator.predict(X)
            for estimator in self.estimators_
        ])

        return predictions

    def predict(self, X):

        predictions = self._predict(X)

        return np.mean(predictions, axis=0)
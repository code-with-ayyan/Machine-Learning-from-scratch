import numpy as np
from copy import deepcopy


class CustomAdaBoostRegressor:

    def __init__(
        self,
        base_estimator,
        n_estimators=50,
        learning_rate=0.1
    ):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):

        self.estimators = []
        self.estimator_weights = []

        X = np.array(X)
        y = np.array(y)

        n_samples = X.shape[0]

        sample_weight = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):

            model = deepcopy(self.base_estimator)

            model.fit(
                X,
                y,
                sample_weight=sample_weight
            )

            y_pred = model.predict(X)

            errors = np.abs(y - y_pred)

            max_error = np.max(errors)

            if max_error == 0:
                self.estimators.append(model)
                self.estimator_weights.append(1.0)
                break

            normalized_errors = errors / max_error

            error = np.sum(
                sample_weight * normalized_errors
            ) / np.sum(sample_weight)

            if error >= 0.5:
                break

            error = np.clip(error, 1e-10, 1 - 1e-10)

            beta = error / (1 - error)

            alpha = self.learning_rate * np.log(1 / beta)

            sample_weight *= np.power(
                beta,
                (1 - normalized_errors)
            )

            sample_weight /= np.sum(sample_weight)

            self.estimators.append(model)
            self.estimator_weights.append(alpha)

        return self

    def predict(self, X):

        if len(self.estimators) == 0:
            raise ValueError("No estimators were trained.")

        X = np.array(X)

        all_predictions = np.array([
            estimator.predict(X)
            for estimator in self.estimators
        ])

        estimator_weights = np.array(
            self.estimator_weights
        )

        predictions = np.average(
            all_predictions,
            axis=0,
            weights=estimator_weights
        )

        return np.array(predictions)
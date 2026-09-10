import numpy as np
from copy import deepcopy


class CustomAdaBoostClassifier:

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

        n_samples = X.shape[0]
        n_classes = len(np.unique(y))

        sample_weight = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):

            model = deepcopy(self.base_estimator)

            model.fit(
                X,
                y,
                sample_weight=sample_weight
            )

            y_pred = model.predict(X)

            incorrect = y_pred != y

            error = np.sum(
                sample_weight * incorrect
            ) / np.sum(sample_weight)

            if error <= 0:
                self.estimators.append(model)
                self.estimator_weights.append(1.0)
                break

            if error >= 1 - (1 / n_classes):
                break

            alpha = self.learning_rate * (
                np.log((1 - error) / error)
                + np.log(n_classes - 1)
            )

            sample_weight *= np.exp(
                alpha * incorrect
            )

            sample_weight /= np.sum(sample_weight)

            self.estimators.append(model)
            self.estimator_weights.append(alpha)

        return self
    
    
    def predict(self, X):

        X = np.array(X)

        all_predictions = np.array([
            estimator.predict(X)
            for estimator in self.estimators
        ])

        predictions = []

        for sample_predictions in all_predictions.T:

            class_scores = {}

            for prediction, alpha in zip(
                sample_predictions,
                self.estimator_weights
            ):
                class_scores[prediction] = (
                    class_scores.get(prediction, 0) + alpha
                )

            final_class = max(
                class_scores,
                key=class_scores.get
            )

            predictions.append(final_class)

        return np.array(predictions)
    
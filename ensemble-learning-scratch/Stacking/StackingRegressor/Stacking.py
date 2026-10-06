import numpy as np
from copy import deepcopy


class StackingRegressor:

    def __init__(
        self,
        estimators,
        meta_estimator,
        k=5
    ):

        self.estimators = estimators
        self.meta_estimator = meta_estimator
        self.k = k

        self.models = []
        self.meta_model = None


    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        indices = np.arange(
            len(X)
        )

        folds = np.array_split(
            indices,
            self.k
        )

        oof_predictions = np.zeros(
            (
                len(X),
                len(self.estimators)
            )
        )

        for validation_indices in folds:

            training_indices = np.setdiff1d(
                indices,
                validation_indices
            )

            X_train_fold = X[
                training_indices
            ]

            y_train_fold = y[
                training_indices
            ]

            X_valid_fold = X[
                validation_indices
            ]

            for j, estimator in enumerate(
                self.estimators
            ):

                model = deepcopy(
                    estimator
                )

                model.fit(
                    X_train_fold,
                    y_train_fold
                )

                predictions = model.predict(
                    X_valid_fold
                )

                oof_predictions[
                    validation_indices,
                    j
                ] = predictions


        self.meta_model = deepcopy(
            self.meta_estimator
        )

        self.meta_model.fit(
            oof_predictions,
            y
        )


        self.models = []

        for estimator in self.estimators:

            model = deepcopy(
                estimator
            )

            model.fit(
                X,
                y
            )

            self.models.append(
                model
            )

        return self


    def predict(self, X):

        X = np.asarray(X)

        predictions = []

        for model in self.models:

            prediction = model.predict(
                X
            )

            predictions.append(
                prediction
            )

        meta_X = np.column_stack(
            predictions
        )

        return self.meta_model.predict(
            meta_X
        )


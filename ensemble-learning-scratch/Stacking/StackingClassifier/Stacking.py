import numpy as np
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold


class CustomStackingClassifier:

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
        self.classes_ = None

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        if len(self.classes_) < 2:
            raise ValueError(
                "At least two classes are required."
            )

        if self.k < 2:
            raise ValueError("k must be at least 2.")

        class_counts = np.unique(
            y, return_counts=True
        )[1]

        if class_counts.min() < self.k:
            raise ValueError(
                "Each class must have at least k samples."
            )

        if not self.estimators:
            raise ValueError(
                "At least one base estimator is required."
            )

        skf = StratifiedKFold(
            n_splits=self.k,
            shuffle=True,
            random_state=42
        )


        n_meta_features = (
            len(self.estimators) * len(self.classes_)
        )

        oof_predictions = np.zeros(
            (len(X), n_meta_features)
        )

        for train_idx, valid_idx in skf.split(X, y):

            X_train = X[train_idx]
            y_train = y[train_idx]

            X_valid = X[valid_idx]

            for j, estimator in enumerate(
                self.estimators
            ):

                model = deepcopy(estimator)

                model.fit(X_train, y_train)

                probabilities = model.predict_proba(
                    X_valid
                )

                aligned_probs = np.zeros(
                    (len(valid_idx), len(self.classes_))
                )

                for col, cls in enumerate(model.classes_):
                    target_col = np.where(
                        self.classes_ == cls
                    )[0][0]

                    aligned_probs[:, target_col] = (
                        probabilities[:, col]
                    )

                start = j * len(self.classes_)
                end = start + len(self.classes_)

                oof_predictions[
                    valid_idx, start:end
                ] = aligned_probs

        self.meta_model = deepcopy(
            self.meta_estimator
        )

        self.meta_model.fit(
            oof_predictions,
            y
        )

        self.models = []

        for estimator in self.estimators:

            model = deepcopy(estimator)
            model.fit(X, y)

            self.models.append(model)

        return self

    def _meta_features(self, X):

        X = np.asarray(X)

        predictions = []

        for model in self.models:

            probabilities = model.predict_proba(X)

            aligned_probs = np.zeros(
                (len(X), len(self.classes_))
            )

            for col, cls in enumerate(model.classes_):

                target_col = np.where(
                    self.classes_ == cls
                )[0][0]

                aligned_probs[:, target_col] = (
                    probabilities[:, col]
                )

            predictions.append(aligned_probs)

        return np.column_stack(predictions)

    def predict(self, X):

        meta_X = self._meta_features(X)

        return self.meta_model.predict(meta_X)

    def predict_proba(self, X):

        meta_X = self._meta_features(X)

        return self.meta_model.predict_proba(meta_X)




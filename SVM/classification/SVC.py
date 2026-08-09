import numpy as np


class CustomSVC:

    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        gamma="scale",
        degree=3,
        coef0=0.0,
        tol=1e-3,
        max_iter=1000,
        class_weight=None,
        random_state=None
    ):

        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alpha = None

        self.b = 0.0

        self.X_train = None
        self.y_train = None

        self.n_samples = None
        self.n_features = None

        self.gamma_val = None

        self.classes_ = None
        self.n_classes = None

        self.C_pos = None
        self.C_neg = None

        self.rng = np.random.default_rng(random_state)

    # =========================================================
    # KERNEL FUNCTIONS
    # =========================================================

    def _kernel_function(self, X1, X2):

        X1 = np.asarray(X1, dtype=np.float64)
        X2 = np.asarray(X2, dtype=np.float64)

        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)

        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)

        if self.kernel == "linear":

            return np.dot(X1, X2.T)

        elif self.kernel == "poly":

            return (
                self.gamma_val * np.dot(X1, X2.T)
                + self.coef0
            ) ** self.degree

        elif self.kernel == "rbf":

            X1_squared = np.sum(
                X1 ** 2,
                axis=1
            ).reshape(-1, 1)

            X2_squared = np.sum(
                X2 ** 2,
                axis=1
            ).reshape(1, -1)

            dist_sq = (
                X1_squared
                + X2_squared
                - 2 * np.dot(X1, X2.T)
            )

            # Numerical safety
            dist_sq = np.maximum(
                dist_sq,
                0
            )

            return np.exp(
                -self.gamma_val * dist_sq
            )

        elif self.kernel == "sigmoid":

            return np.tanh(
                self.gamma_val
                * np.dot(X1, X2.T)
                + self.coef0
            )

        else:

            raise ValueError(
                f"Unknown kernel: {self.kernel}"
            )

    # =========================================================
    # KERNEL MATRIX
    # =========================================================

    def _compute_kernel_matrix(self, X):

        return self._kernel_function(
            X,
            X
        )

    # =========================================================
    # C VALUE FOR EACH SAMPLE
    # =========================================================

    def _get_C(self, i):

        i = np.asarray(i)

        result = np.where(
            self.y_train[i] == 1,
            self.C_pos,
            self.C_neg
        )

        if result.ndim == 0:
            return float(result)

        return result

    # =========================================================
    # TRAINING DECISION FUNCTION
    # =========================================================

    def _decision_function_training(
        self,
        i,
        K
    ):

        return (
            np.sum(
                self.alpha
                * self.y_train
                * K[:, i]
            )
            + self.b
        )

    # =========================================================
    # SELECT SECOND ALPHA
    # =========================================================

    def _select_second_alpha(
        self,
        i,
        E_i,
        K
    ):

        indices = np.arange(
            self.n_samples
        )

        C_values = self._get_C(
            indices
        )

        # -----------------------------------------------------
        # 1. Prefer non-bound alphas
        # -----------------------------------------------------

        non_bound = np.where(
            (self.alpha > 0)
            &
            (self.alpha < C_values)
        )[0]

        non_bound = non_bound[
            non_bound != i
        ]

        if len(non_bound) > 0:

            E_non_bound = np.array(
                [
                    self._decision_function_training(
                        j,
                        K
                    )
                    - self.y_train[j]

                    for j in non_bound
                ]
            )

            best_index = np.argmax(
                np.abs(
                    E_non_bound - E_i
                )
            )

            return int(
                non_bound[best_index]
            )

        # -----------------------------------------------------
        # 2. Try any alpha > 0
        # -----------------------------------------------------

        candidates = np.where(
            self.alpha > 0
        )[0]

        candidates = candidates[
            candidates != i
        ]

        if len(candidates) > 0:

            return int(
                self.rng.choice(
                    candidates
                )
            )

        # -----------------------------------------------------
        # 3. IMPORTANT:
        # At beginning all alpha = 0.
        # Select any other sample.
        # -----------------------------------------------------

        candidates = np.arange(
            self.n_samples
        )

        candidates = candidates[
            candidates != i
        ]

        if len(candidates) > 0:

            return int(
                self.rng.choice(
                    candidates
                )
            )

        return -1

    # =========================================================
    # FIT
    # =========================================================

    def fit(self, X, y):

        X = np.asarray(
            X,
            dtype=np.float64
        )

        y = np.asarray(
            y,
            dtype=np.float64
        ).flatten()

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        if X.ndim != 2:

            raise ValueError(
                "X must be a 2D array."
            )

        if len(X) != len(y):

            raise ValueError(
                "Number of samples in X and y must be equal."
            )

        if np.isnan(X).any():

            raise ValueError(
                "X contains NaN values."
            )

        if np.isinf(X).any():

            raise ValueError(
                "X contains infinite values."
            )

        # -----------------------------------------------------
        # Store training data
        # -----------------------------------------------------

        self.X_train = X
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        # -----------------------------------------------------
        # Original classes
        # -----------------------------------------------------

        self.classes_ = np.unique(y)

        self.n_classes = len(
            self.classes_
        )

        if self.n_classes != 2:

            raise ValueError(
                "CustomSVC currently supports "
                "binary classification only."
            )

        # -----------------------------------------------------
        # Convert labels to -1 and +1
        # -----------------------------------------------------

        self.y_train = np.where(
            y == self.classes_[0],
            -1.0,
            1.0
        )

        # -----------------------------------------------------
        # Gamma
        # -----------------------------------------------------

        if self.gamma == "scale":

            variance = np.var(
                self.X_train
            )

            if variance == 0:

                self.gamma_val = 1.0

            else:

                self.gamma_val = (
                    1.0
                    /
                    (
                        self.n_features
                        * variance
                    )
                )

        elif self.gamma == "auto":

            self.gamma_val = (
                1.0
                /
                self.n_features
            )

        else:

            self.gamma_val = float(
                self.gamma
            )

        # -----------------------------------------------------
        # Class weights
        # -----------------------------------------------------

        if self.class_weight == "balanced":

            total_samples = len(
                self.y_train
            )

            class_weights = {}

            for cls in [-1, 1]:

                class_count = np.sum(
                    self.y_train == cls
                )

                if class_count == 0:

                    class_weights[cls] = 1.0

                else:

                    class_weights[cls] = (
                        total_samples
                        /
                        (
                            2
                            * class_count
                        )
                    )

            self.C_pos = (
                self.C
                * class_weights[1]
            )

            self.C_neg = (
                self.C
                * class_weights[-1]
            )

        else:

            self.C_pos = self.C
            self.C_neg = self.C

        # -----------------------------------------------------
        # Kernel matrix
        # -----------------------------------------------------

        K = self._compute_kernel_matrix(
            self.X_train
        )

        # -----------------------------------------------------
        # Initialize alpha
        # -----------------------------------------------------

        self.alpha = np.zeros(
            self.n_samples,
            dtype=np.float64
        )

        self.b = 0.0

        # =====================================================
        # SMO TRAINING
        # =====================================================

        for iteration in range(
            self.max_iter
        ):

            alpha_previous = (
                self.alpha.copy()
            )

            num_changed = 0

            for i in range(
                self.n_samples
            ):

                # -------------------------------------------------
                # Error
                # -------------------------------------------------

                E_i = (
                    self._decision_function_training(
                        i,
                        K
                    )
                    -
                    self.y_train[i]
                )

                C_i = self._get_C(i)

                # -------------------------------------------------
                # Check KKT violation
                # -------------------------------------------------

                violates_kkt = (

                    (
                        self.y_train[i]
                        * E_i
                        < -self.tol
                    )
                    and
                    (
                        self.alpha[i]
                        < C_i
                    )

                ) or (

                    (
                        self.y_train[i]
                        * E_i
                        > self.tol
                    )
                    and
                    (
                        self.alpha[i]
                        > 0
                    )

                )

                if not violates_kkt:

                    continue

                # -------------------------------------------------
                # Select j
                # -------------------------------------------------

                j = self._select_second_alpha(
                    i,
                    E_i,
                    K
                )

                if j == -1:

                    continue

                # -------------------------------------------------
                # Error of j
                # -------------------------------------------------

                E_j = (
                    self._decision_function_training(
                        j,
                        K
                    )
                    -
                    self.y_train[j]
                )

                alpha_i_old = (
                    self.alpha[i]
                )

                alpha_j_old = (
                    self.alpha[j]
                )

                C_j = self._get_C(j)

                # -------------------------------------------------
                # Calculate bounds L and H
                # -------------------------------------------------

                if (
                    self.y_train[i]
                    !=
                    self.y_train[j]
                ):

                    L = max(
                        0.0,
                        self.alpha[j]
                        -
                        self.alpha[i]
                    )

                    H = min(
                        C_j,
                        C_j
                        +
                        self.alpha[j]
                        -
                        self.alpha[i]
                    )

                else:

                    L = max(
                        0.0,
                        self.alpha[i]
                        +
                        self.alpha[j]
                        -
                        C_i
                    )

                    H = min(
                        C_i,
                        self.alpha[i]
                        +
                        self.alpha[j]
                    )

                if L >= H:

                    continue

                # -------------------------------------------------
                # Eta
                # -------------------------------------------------

                eta = (
                    2.0 * K[i, j]
                    -
                    K[i, i]
                    -
                    K[j, j]
                )

                if eta >= 0:

                    continue

                # -------------------------------------------------
                # Update alpha_j
                # -------------------------------------------------

                self.alpha[j] -= (
                    self.y_train[j]
                    *
                    (E_i - E_j)
                    /
                    eta
                )

                self.alpha[j] = np.clip(
                    self.alpha[j],
                    L,
                    H
                )

                # -------------------------------------------------
                # Check if change is significant
                # -------------------------------------------------

                if (
                    abs(
                        self.alpha[j]
                        -
                        alpha_j_old
                    )
                    < 1e-5
                ):

                    continue

                # -------------------------------------------------
                # Update alpha_i
                # -------------------------------------------------

                self.alpha[i] += (

                    self.y_train[i]
                    *
                    self.y_train[j]
                    *
                    (
                        alpha_j_old
                        -
                        self.alpha[j]
                    )

                )

                # -------------------------------------------------
                # Calculate b1 and b2
                # -------------------------------------------------

                b1 = (

                    self.b
                    -
                    E_i
                    -
                    self.y_train[i]
                    *
                    (
                        self.alpha[i]
                        -
                        alpha_i_old
                    )
                    *
                    K[i, i]

                    -
                    self.y_train[j]
                    *
                    (
                        self.alpha[j]
                        -
                        alpha_j_old
                    )
                    *
                    K[i, j]

                )

                b2 = (

                    self.b
                    -
                    E_j
                    -
                    self.y_train[i]
                    *
                    (
                        self.alpha[i]
                        -
                        alpha_i_old
                    )
                    *
                    K[i, j]

                    -
                    self.y_train[j]
                    *
                    (
                        self.alpha[j]
                        -
                        alpha_j_old
                    )
                    *
                    K[j, j]

                )

                # -------------------------------------------------
                # Update bias
                # -------------------------------------------------

                if (
                    0
                    <
                    self.alpha[i]
                    <
                    C_i
                ):

                    self.b = b1

                elif (
                    0
                    <
                    self.alpha[j]
                    <
                    C_j
                ):

                    self.b = b2

                else:

                    self.b = (
                        b1 + b2
                    ) / 2.0

                num_changed += 1

            # -----------------------------------------------------
            # Check convergence
            # -----------------------------------------------------

            max_change = np.max(
                np.abs(
                    self.alpha
                    -
                    alpha_previous
                )
            )

            if (
                max_change < self.tol
                and
                num_changed == 0
            ):

                break

        # =========================================================
        # SUPPORT VECTORS
        # =========================================================

        support_mask = (
            self.alpha > self.tol
        )

        self.support_vectors = (
            self.X_train[
                support_mask
            ]
        )

        self.support_vector_labels = (
            self.y_train[
                support_mask
            ]
        )

        self.support_vector_alpha = (
            self.alpha[
                support_mask
            ].copy()
        )

        # ---------------------------------------------------------
        # Safety check
        # ---------------------------------------------------------

        if len(
            self.support_vectors
        ) == 0:

            raise RuntimeError(
                "SVM training produced no "
                "support vectors. Try increasing "
                "C or max_iter."
            )

        return self

    # =========================================================
    # DECISION FUNCTION
    # =========================================================

    def decision_function(self, X):

        X = np.asarray(
            X,
            dtype=np.float64
        )

        if X.ndim == 1:

            X = X.reshape(
                1,
                -1
            )

        if X.shape[1] != self.n_features:

            raise ValueError(
                f"Expected {self.n_features} "
                f"features, got {X.shape[1]}."
            )

        # -----------------------------------------------------
        # IMPORTANT:
        # Use only support vectors.
        # -----------------------------------------------------

        K_test = self._kernel_function(
            X,
            self.support_vectors
        )

        decision = (

            np.dot(
                K_test,
                self.support_vector_alpha
                *
                self.support_vector_labels
            )

            +
            self.b

        )

        return decision

    # =========================================================
    # PREDICT
    # =========================================================

    def predict(self, X):

        decision = (
            self.decision_function(X)
        )

        return np.where(
            decision >= 0,
            self.classes_[1],
            self.classes_[0]
        )
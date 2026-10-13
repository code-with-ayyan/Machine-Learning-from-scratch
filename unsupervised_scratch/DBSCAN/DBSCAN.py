
import numpy as np


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        X = np.asarray(X, dtype=float)

        n_samples = len(X)

        labels = np.full(n_samples, -2, dtype=int)

        cluster_id = 0

        for point_idx in range(n_samples):

            if labels[point_idx] != -2:
                continue

            neighbors = self._region_query(X, point_idx)

            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1
                continue

       
            labels[point_idx] = cluster_id

            seeds = list(neighbors)
            seed_set = set(seeds)

            i = 0

            while i < len(seeds):
                neighbor_idx = seeds[i]

             
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id

                elif labels[neighbor_idx] == -2:
                    labels[neighbor_idx] = cluster_id

                    new_neighbors = self._region_query(
                        X, neighbor_idx
                    )

          
                    if len(new_neighbors) >= self.min_samples:
                        for new_idx in new_neighbors:
                            if new_idx not in seed_set:
                                seeds.append(new_idx)
                                seed_set.add(new_idx)

                i += 1

            cluster_id += 1

        self.labels_ = labels
        return labels

    def _region_query(self, X, point_idx):

        distances = np.linalg.norm(
            X - X[point_idx], axis=1
        )

        return np.where(distances <= self.eps)[0].tolist()



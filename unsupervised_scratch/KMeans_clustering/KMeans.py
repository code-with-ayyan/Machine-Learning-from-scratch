import random 
import numpy as np 


class CustomKMeans:
    
    def __init__(self, n_clusters= 2, max_iter= 100):
        
        self.n_clusters = n_clusters
        self.max_iter   = max_iter
        
        self.centroids  = None 
        self.inertia_score    = None
        
        
    def fit_predict(self, X):
        
        
        X = np.asarray(X)
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]
        
        for i in range(self.max_iter):
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids.copy()
            
            self.centroids = self.move_centroids(X, cluster_group)
            
            if (old_centroids == self.centroids).all():
                break
                
        
        self.inertia_score = self.inertia(X, cluster_group)
        return cluster_group

            
            
    def assign_clusters(self, X):
        
        cluster_group = []
        distances = []

        for row in X:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            cluster_group.append(index_pos)
            distances.clear()

        return np.array(cluster_group)
    
    
    def move_centroids(self,X,cluster_group):
        
        new_centroids = []

        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis=0))

        return np.array(new_centroids)
    
    
    def inertia(self, X, cluster_group):
        
        X = np.asarray(X)
        total_inertia = 0
        
       
        for type in np.unique(cluster_group):
            
            cluster_points = X[cluster_group == type]
           
            centroid = self.centroids[type]
            
            total_inertia += np.sum((cluster_points - centroid) ** 2)
            
        return total_inertia
    
    def inertia_(self):
        
        return self.inertia_score
        
        


        
        
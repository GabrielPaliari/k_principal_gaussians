'''
IMPORTS
'''
import numpy as np
from pre_processing import reduce_dim_PCA
from gaussian_mixture import fit_gaussian_mixture
from segments_generator import create_segs_from_model
from search_optimization import create_hnsw_from_centroids, find_closest_segments
from MetaParams import MetaParams, default_meta_params
from model.distance import euclidean_distance_to_segment
from tqdm import tqdm   

class KPrincipalGaussiansClassifier:
    def __init__(self, meta_params=default_meta_params):
        self.meta_params=meta_params
        self.all_centroids = []
        self.all_segments = []
        self.all_pc2_variances = []
        self.all_weights = []
        self.num_classes=meta_params.num_classes

    def fit(self, train_data, labels):
        self.train_data=train_data
        self.labels=labels
        
        pca, k_segments_per_class = self.__reduce_dim()
        self.pca=pca
        self.k_segments_per_class=k_segments_per_class
        
        self.__fit_gaussian_mixture_models()
        self.__create_search_optimization()
    
    def __reduce_dim(self):
        # Reduce dim. with PCA
        pca, k_segments_per_class = reduce_dim_PCA(self.train_data, self.meta_params)
        print(f"PCA k_seg_por_classe: {k_segments_per_class}")
        return pca, k_segments_per_class

    def __fit_gaussian_mixture_models(self):
        all_centroids = []
        all_segments = []
        all_pc2_variances = []
        all_weights = []
        for class_idx in range(self.num_classes):
            print(f"Treinando GaussianMixture para a classe: {class_idx}")

            class_data = self.train_data[self.labels == class_idx]
            processed_data = self.pca.transform(class_data)
            gmm = fit_gaussian_mixture(data=processed_data, k_segments_per_class=self.k_segments_per_class, meta_params=self.meta_params)
            
            centroids = gmm.means_
            weights = gmm.weights_
            segments, pc2_variances = create_segs_from_model(centroids, gmm.covariances_, meta_params=self.meta_params)
            
            all_centroids.extend(centroids)
            all_weights.extend(weights)
            all_segments.extend(segments)
            all_pc2_variances.extend(pc2_variances)
            
        self.all_weights = np.array(all_weights)
        self.all_centroids = np.array(all_centroids)
        self.all_segments = np.array(all_segments)
        self.all_pc2_variances = np.array(all_pc2_variances)
    
    def __create_search_optimization(self):
        self.hnsw = create_hnsw_from_centroids(all_centroids=self.all_centroids, meta_params=self.meta_params)
    
    def predict(self, points, weight_pow=0.5, variance_pow=0.5):      
        transformed_points = self.pca.transform(points)
        predictions = np.zeros(points.shape[0])
        for idx, point in tqdm(enumerate(transformed_points), ncols=80):
            closest_segs_idx = find_closest_segments(hnsw=self.hnsw, point=point, metaparams=self.meta_params)        
            best_index = self.__validate_segments_with_weights(point=point, indices=closest_segs_idx, weight_pow=weight_pow, variance_pow=variance_pow)
            class_predicted = best_index // (self.k_segments_per_class)
            predictions[idx] = class_predicted
        return predictions.astype(int) 
        
    def __validate_segments_with_weights(self, point, indices, variance_pow=0.5):
        """
        Valida os segmentos considerando as distâncias ajustadas com PCA2 e os pesos das gaussianas.
        
        segment_weights: array contendo os pesos das gaussianas associadas aos segmentos.
        """
        min_adjusted_distance = float('inf')
        best_index = -1

        for idx in indices:
            A, B = self.all_segments[idx]
            distance = euclidean_distance_to_segment(P=point, A=A, B=B)
            
            # Ajustar distância com PCA2 e peso gaussiano
            variance_weight = self.all_pc2_variances[idx] ** variance_pow
            adjusted_distance = distance / variance_weight
            
            if adjusted_distance < min_adjusted_distance:
                min_adjusted_distance = adjusted_distance
                best_index = idx

        return best_index
        
        
        




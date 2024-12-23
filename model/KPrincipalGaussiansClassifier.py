'''
IMPORTS
'''
import numpy as np
from pre_processing import reduce_dim_PCA
from gaussian_mixture import fit_gaussian_mixture
from segments_generator import create_segs_from_model
from search_optimization import create_hnsw_from_centroids, find_closest_segments
from MetaParams import MetaParams, default_meta_params
from tqdm import tqdm   
from distance import mahalanobis_distance
class KPrincipalGaussiansClassifier:
    def __init__(self, meta_params=default_meta_params):
        self.meta_params=meta_params
        self.all_means = []
        self.all_precisions = []
        self.all_indexes_to_exclude = []
        self.num_classes=meta_params.num_classes
        self.class_components_weights=meta_params.class_components_weights

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
        all_means = []
        all_precisions = []
        all_indexes_to_exclude = []
        for class_idx in range(self.num_classes):
            print(f"Treinando GaussianMixture para a classe: {class_idx}")
            original_class_data = self.train_data[self.labels == class_idx]
            
            "Note que são 2 PCAs diferentes, um global e um da classe, usado para desconsiderar componentes das gaussianas"
            class_pca, ideal_k_class_specific = reduce_dim_PCA(original_class_data, self.meta_params)
            print(f"K componentes ideal para classe {class_idx}: {ideal_k_class_specific}")
            ideal_k_class_specific = ideal_k_class_specific * self.class_components_weights[class_idx]
            
            processed_class_data = self.pca.transform(original_class_data)
            class_pca.fit(original_class_data)
            
            means, precisions, indexes_to_exclude = fit_gaussian_mixture(class_data=processed_class_data, k_segments_per_class=self.k_segments_per_class, ideal_k_class_specific=ideal_k_class_specific, meta_params=self.meta_params)
            
            all_means.extend(means)
            all_precisions.extend(precisions)
            all_indexes_to_exclude.extend(indexes_to_exclude + class_idx * self.k_segments_per_class)
        self.all_means = np.array(all_means)
        self.all_precisions = np.array(all_precisions)
        self.all_indexes_to_exclude = np.array(indexes_to_exclude)
        
    def __create_search_optimization(self):
        self.hnsw = create_hnsw_from_centroids(
            all_means=self.all_means,
            excluded_indexes=self.all_indexes_to_exclude,
            meta_params=self.meta_params
        )
    
    def predict(self, points):      
        transformed_points = self.pca.transform(points)
        predictions = np.zeros(points.shape[0])
        for idx  in tqdm(range(len(transformed_points)), ncols=80):
            point = transformed_points[idx]
            closest_segs_idx = find_closest_segments(hnsw=self.hnsw, point=point, metaparams=self.meta_params)        
            best_index = self.__validate_segments_with_weights(point=point, indices=closest_segs_idx)
            class_predicted = best_index // (self.k_segments_per_class)
            predictions[idx] = class_predicted
        return predictions.astype(int) 
        
    def __validate_segments_with_weights(self, point, indices):
        """
        Valida os segmentos considerando as distâncias ajustadas com PCA2 e os pesos das gaussianas.
        
        segment_weights: array contendo os pesos das gaussianas associadas aos segmentos.
        """
        min_distance = float('inf')
        best_index = -1

        for idx in indices:
            mean = self.all_means[idx]
            prec = self.all_precisions[idx]
            distance = mahalanobis_distance(x=point, mean=mean, precision=prec)
            
            if distance < min_distance:
                min_distance = distance
                best_index = idx

        return best_index
        
        
        




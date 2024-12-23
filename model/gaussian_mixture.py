from sklearn.mixture import BayesianGaussianMixture
from MetaParams import default_meta_params
import numpy as np

def fit_gaussian_mixture(class_data, k_segments_per_class=100, ideal_k_class_specific=50, meta_params=default_meta_params):
    """
    Approximate a principal curve using a Gaussian Mixture Model.
    
    Parameters:
    - data: np.ndarray, shape (n_samples, n_features)
      Input data.
    - n_components: int
      Number of Gaussian components in the GMM.
    - n_points: int
      Number of points to sample on the principal curve.

    Returns:
    - curve: np.ndarray, shape (n_points, n_features)
      The computed principal curve.
    """
    gmm = BayesianGaussianMixture(
        n_components=k_segments_per_class,
        max_iter=meta_params.max_iter,
        covariance_type=meta_params.covariance_type,
        weight_concentration_prior_type=meta_params.weight_concentration_prior_type,
        tol=meta_params.tol,
        random_state=meta_params.random_state,
        verbose=meta_params.verbose, 
        init_params=meta_params.init_params,
        n_init=meta_params.n_init,
        warm_start=meta_params.warm_start,
    )
    gmm.fit(class_data)
    
    means = gmm.means_
    precisions = gmm.precisions_
    weights = gmm.weights_
    idx_weigths_desc = np.argsort(weights)[::-1]
    means_sorted = means[idx_weigths_desc]
    precisions_sorted = precisions[idx_weigths_desc]
    
    num_indexes_to_exclude = k_segments_per_class - ideal_k_class_specific
    indexes_to_exclude = []
    if (num_indexes_to_exclude > 0):
      indexes_to_exclude = idx_weigths_desc[-int(num_indexes_to_exclude):]
    
    return means_sorted, precisions_sorted, indexes_to_exclude
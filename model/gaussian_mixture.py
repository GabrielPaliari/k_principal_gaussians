from sklearn.mixture import BayesianGaussianMixture
from MetaParams import default_meta_params
def fit_gaussian_mixture(data, k_segments_per_class=100, meta_params=default_meta_params):
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
    gaussian_mixture_model = BayesianGaussianMixture(
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
    gaussian_mixture_model.fit(data)
    return gaussian_mixture_model
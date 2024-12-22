from dataclasses import dataclass
from enum import Enum

class MetricType(Enum):
    EUCLIDEAN = 'l2'
    INNER_PRODUCT = 'ip'
    COSINE = 'cosine'

@dataclass
class MetaParams:
    # Global
    num_classes: int = 10

    # PCA pre processing
    variance_perc_threshold: float = 0.95

    # Bayesian Gaussian Mixture
    max_iter: int = 100
    tol: float = 1e-3
    covariance_type: str = "full"
    weight_concentration_prior_type: str = "dirichlet_process"
    init_params: str = "k-means++"
    random_state: int = 2
    warm_start: bool = True
    n_init: int = 2
    verbose: int = 2
    norm_std_factor: float = 2.0

    # HNSW
    ef: int = 500
    M: int = 16
    metric_type: MetricType = MetricType.EUCLIDEAN
    k_near_centroids: int = 10

    def __iter__(self):
        # Itera apenas sobre as propriedades anotadas
        return iter([getattr(self, key) for key in self.__annotations__.keys()])


default_meta_params = MetaParams()
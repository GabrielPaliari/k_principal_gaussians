def euclidean_distance_to_segment(P, A, B):
    """
    Calculate the Euclidean distance from point P to a line segment AB.
    
    Parameters:
    - P: numpy array of shape (n,), the point
    - A: numpy array of shape (n,), one endpoint of the segment
    - B: numpy array of shape (n,), the other endpoint of the segment
    
    Returns:
    - distance: float, the Euclidean distance from P to segment AB
    """
    import numpy as np
    AB = B - A
    AP = P - A
    
    # Projection factor t
    t = np.dot(AP, AB) / np.dot(AB, AB)
    
    # Clamp t to [0, 1]
    t_clamped = max(0, min(1, t))
    
    # Closest point on segment
    Q = A + t_clamped * AB
    
    # Distance from P to Q
    distance = np.linalg.norm(P - Q)
    return distance

def mahalanobis_distance(x, mean, precision):
    import numpy as np
    """
    Calcula a distância de Mahalanobis de um ponto para uma distribuição gaussiana.
    
    Args:
    - x: Ponto (array 1D).
    - mean: Média da gaussiana (array 1D).
    - cov: Matriz de covariância (array 2D).
    
    Retorna:
    - Distância de Mahalanobis (float).
    """
    delta = x - mean
    dist = np.sqrt(np.dot(np.dot(delta.T, precision), delta))
    return dist
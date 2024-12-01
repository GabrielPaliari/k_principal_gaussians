import numpy as np
from sklearn.decomposition import PCA
import time

def create_line_segment(center, direction, length):
    """
    Create a line segment with mean in point A, direction D, and length L.

    Parameters:
        A (array-like): The mean of the line segment (e.g., [x, y]).
        D (array-like): The direction of the line segment (e.g., [dx, dy]).
        L (float): The length of the line segment.

    Returns:
        tuple: Two points representing the endpoints of the line segment.
    """
    # Convert inputs to NumPy arrays
    center = np.array(center)
    direction = np.array(direction)

    # Normalize the direction vector
    D_normalized = direction / np.linalg.norm(direction)


    # Calculate the endpoint
    startpoint = center - length / 2 * D_normalized    
    endpoint = center + length / 2 * D_normalized

    # Return the two endpoints of the line segment
    return np.array([startpoint, endpoint])

def distance_point_to_segment(point, segment):
    # Descompactar coordenadas
    px, py = point
    segment_start, segment_end = segment
    x1, y1 = segment_start
    x2, y2 = segment_end

    # Vetores
    AP = np.array([px - x1, py - y1])
    AB = np.array([x2 - x1, y2 - y1])
    
    # Comprimento ao quadrado do segmento
    AB_length_squared = AB.dot(AB)
    
    if AB_length_squared == 0:
        # A e B são o mesmo ponto
        return np.linalg.norm(AP)
    
    # Projeção escalar
    t = AP.dot(AB) / AB_length_squared

    # Restringir t ao intervalo [0, 1] (será que se colocarmos uma margem ou um ruído para permitir que as linhas crescam e diminuam, dá bom?)
    if (t < 0 or t > 1): 
        return float('inf')
    
    # Ponto projetado no segmento
    projection = np.array([x1, y1]) + t * AB
    
    # Distância euclidiana entre P e a projeção
    distance = np.linalg.norm(np.array([px, py]) - projection)
    
    return distance

def get_new_segment(data, segments):
    if (len(segments) == 0):
        return update_single_segment(data)
    
    min_total_distance = float('inf')
    best_point = None
    min_distances = []
    for point in data:
        segments_distances = [distance_point_to_segment(point, segment) for segment in segments]
        min_distances.append([point, min(segments_distances)])
    
    start_time = time.time()
    for candidate_point in data:        
        total_distance = 0
        
        for point_dist in min_distances:            
            point, dist = point_dist
            point_to_candidate_dist = np.linalg.norm(point - candidate_point) 
            total_distance += min(dist, point_to_candidate_dist)
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_point = candidate_point
    print(f"Time to calc best point: {time.time() - start_time:.4f} seconds")
    
    # Return the best point as the new segment
    return np.array([best_point, best_point])    
    
def update_single_segment(cluster_points):        
    pca = PCA(n_components=1)
    pca.fit(cluster_points)
    direction = pca.components_[0]
    std_dev = np.sqrt(pca.explained_variance_[0])           
    length = 3 / 2 * std_dev 
    mean = cluster_points.mean(axis=0)
    return create_line_segment(mean, direction, length)              
    
def assign_points_to_segments(data, segments):
    assignments = []
    for point in data:
        distances = [distance_point_to_segment(point, segment) for segment in segments]
        assignments.append(np.argmin(distances))
    return assignments

def update_segments(data, assignments, k):
    segments = []
    for i in range(k):
        cluster_points = data[np.array(assignments) == i]
        if len(cluster_points) > 0:
            segment = update_single_segment(cluster_points)
            segments.append(segment)  
                  
    return segments

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def find_closest_point(current_point, points):
    """Find the closest point to current_point among points, excluding points in exclude_points."""
    min_distance = float('inf')
    closest_point = None
    for point in points:  # Convert to tuple for comparison
        distance = calculate_distance(current_point, point)
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    return closest_point, min_distance

def plot_segments(segments, plt, isConnection):
    """
    Plots a line between each consecutive pair of points in a NumPy array.

    Parameters:
        points (numpy.ndarray): A 2D array of shape (n, 2), where each row is a point [x, y].
    """
    color='c'
    width=5
    if (isConnection):
        color='r'
        width=1
        
    for i in range(len(segments)):
        # Extract consecutive pairs of points
        start, end = segments[i]

        # Plot the line between the points
        plt.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=width, label=f"Segment {i}")

def k_segments_algorithm(data, k_max=5):    
    segments = [get_new_segment(data, [])]
    history = [[segments, [], "k = 1"]]
    for k in range(2, k_max + 1):
        new_seg = get_new_segment(data, segments)
        segments.append(new_seg)
        
        voronoi_regions = assign_points_to_segments(data, segments)
                
        segments = update_segments(data, voronoi_regions, k)
                
        history.append([segments, "k = " + str(k)])
    return segments, history

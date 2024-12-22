
import numpy as np
from MetaParams import default_meta_params

def create_segs_from_model(means, covariances, meta_params=default_meta_params):
    """
    Cria os segmentos e as direções de segunda maior variação a partir das médias e covariâncias
    
    gmm: Bayesian Gaussian Mixture Model 
    """
    pc2_variances = []
    segments = []
    for i, mean in enumerate(means):
        # Compute PCA on the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariances[i])
        
        # Ordenar os índices dos valores em ordem decrescente
        sorted_vals_idx = np.argsort(eigenvalues)[::-1]
        
        # Pegue os dois maiores autovalores
        val_index1, val_index2 = sorted_vals_idx[:2]
        eig_val1, eig_val2 = np.sqrt(eigenvalues[val_index1]), eigenvalues[val_index2]

        eig_vec1 = np.argmax(eigenvectors)
                
        segment = create_segment(mean, eig_vec1, eig_val1, meta_params=meta_params)
        segments.append(segment)
        pc2_variances.append(eig_val2)
    return np.array(segments), np.array(pc2_variances)

def create_segment(mean, eigenvector, eigenvalue, meta_params=default_meta_params):
    '''
    norm_std_factor: Multiplica o autovalor do 1° CP. considerando dados normais: 
        68% dos dados estão dentro de ±1*std_dev.
        95% dos dados estão dentro de ±2*std_dev.
        99.7% dos dados estão dentro de ±3*std_dev.
        Também pode-se usar o intervalo interquartil: iqr = 1.348 * std_dev
    '''
    norm_std_factor = meta_params.norm_std_factor
    p1 = mean + eigenvector * eigenvalue * norm_std_factor
    p2 = mean - eigenvector * eigenvalue * norm_std_factor
    return np.array([p1, p2])

'''
Se for necessário ordenar e conectar os segmentos, descomentar essa função abaixo
Não iremos utilizar num primeiro momento para simplificar o algoritmo

'''
# def order_segments(segments, ortogonal_components):
#     '''
#     Ordena Segmentos e também a componente principal ortogonal  
#     '''
#     # Lista para armazenar a ordem dos segmentos
#     ordered_segments = []

#     # Usar o primeiro segmento como ponto inicial
#     current_segment = segments[0]
#     ordered_segments.append(current_segment)
#     ordered_pcs = [ortogonal_components[0]]
#     current_start, current_end = current_segment

#     # Criar uma lista dos segmentos restantes
#     remaining_segments = list(segments[1:])
#     remaining_pcs = list(ortogonal_components[1:])


#     while remaining_segments:
#         # Criar uma lista com todos os extremos dos segmentos restantes
#         candidates = []
#         for seg in remaining_segments:
#             candidates.extend(seg)
#         candidates = np.array(candidates)
        
#         # Calcular a menor distância entre o ponto final do segmento atual e os candidatos
#         distances_end = cdist([current_end], candidates)
#         distances_start = cdist([current_start], candidates)
#         min_idx_end = np.argmin(distances_end)
#         min_dist_end = np.min(distances_end)
#         min_idx_start = np.argmin(distances_start)
#         min_dist_start = np.min(distances_start)
        
#         if min_dist_start > min_dist_end:  
#             next_segment_idx = min_idx_end // 2
#             next_segment = remaining_segments[next_segment_idx]
#             ordered_pcs.append(remaining_pcs[next_segment_idx])
#             if min_idx_end % 2 == 0:
#                 current_end = next_segment[1] 
#                 ordered_segments.append(next_segment)
#             else:
#                 current_end = next_segment[0] 
#                 ordered_segments.append([next_segment[1], next_segment[0]])
                
#         else:
#             next_segment_idx = min_idx_start // 2
#             next_segment = remaining_segments[next_segment_idx]
#             ordered_pcs.insert(0, remaining_pcs[next_segment_idx])
#             if min_idx_start % 2 == 0:
#                 current_start = next_segment[1] 
#                 ordered_segments.insert(0, [next_segment[1], next_segment[0]])
#             else:
#                 current_start = next_segment[0] 
#                 ordered_segments.insert(0, next_segment)
            
#         # Remover o segmento já utilizado
#         del remaining_segments[next_segment_idx]
#         del remaining_pcs[next_segment_idx]

#     return np.array(ordered_segments), ordered_pcs

# def calc_connection_seg(seg_prev, pc_prev, seg_current, pc_current):
#     pc_norm = (np.linalg.norm(pc_current) + np.linalg.norm(pc_prev)) / 2
#     return {
#         'max_dist': pc_norm,
#         'seg_points': np.array([seg_prev[1], seg_current[0]]),
#         'is_conn': True
#     }   
      
# def extract_final_curve(ordered_segments, ordered_pcs):
#     final_curve = []
#     for i, segment in enumerate(ordered_segments):                
#         final_curve.append({
#             'ortogonal_pc': ordered_pcs[i],
#             'seg_points': segment,
#             'is_conn': False
#         }) 
#         if (i>=1 and i < len(ordered_segments)):
#             conn_seg = calc_connection_seg(ordered_segments[i-1], ordered_pcs[i-1], segment, ordered_pcs[i])        
#             final_curve.append(conn_seg)
#     return final_curve
        
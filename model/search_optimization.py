import hnswlib
from MetaParams import default_meta_params
import numpy as np
    
def create_hnsw_from_centroids(all_centroids, meta_params):
    """
    Cria uma HNSW (Hierarchical Navigable Small World) para busca aproximada.
    centroids: array de forma (n, d), onde n é o número de segmentos e d é a dimensionalidade.
    ef: parâmetro de controle de precisão durante a busca (maior é mais preciso).
    M: número de conexões por nó na estrutura HNSW.

    Retorna:
    - hnsw: índice HNSW criado a partir dos centroides dos segmentos.
    """
    num_elements, dim = all_centroids.shape

    # Inicializar índice HNSW
    hnsw = hnswlib.Index(space='l2', dim=dim)
    hnsw.init_index(
        max_elements=num_elements, 
        ef_construction=meta_params.ef, 
        M=meta_params.M,
    )

    # Adicionar os centroides ao índice
    hnsw.add_items(all_centroids)

    # Ajustar ef para busca
    hnsw.set_ef(meta_params.ef)

    return hnsw

def find_closest_segments(hnsw, point, metaparams=default_meta_params):
    """
    Encontra os segmentos mais próximos de um ponto com base nos seus centroides usando HNSW.
    hnsw: índice HNSW criado a partir dos centroides. 
    point: array de forma (d,), representando o ponto de interesse.
    k: número de segmentos candidatos a retornar. 
        Note que a busca utiliza os centroides como referência, o que diminui a precisão. A projeção a cada segmento será calculada depois

    Retorna:
    - indices: índices dos segmentos mais próximos.
    """
    k_near_centroids=metaparams.k_near_centroids
    indices, _ = hnsw.knn_query(point, k=k_near_centroids)  # Busca os k centroides mais próximos
    return indices.reshape(-1)

# Atualmente não iremos utilizar este algoritmo, mas pode ser usado para encontrar o k ideal para cada região
def determine_k_segments_dinamically(hnsw, point, max_k=50, cv_threshold=0.2):
    """
    Determina o número ideal de segmentos mais próximos com base na variabilidade (CV) das distâncias.
    
    hnsw: índice HNSW criado a partir dos centroides.
    point: array representando o ponto de interesse.
    max_k: número máximo de segmentos candidatos.
    cv_threshold: limite para o coeficiente de variação (CV) das distâncias.
    
    OBS: Se não demorar muito, usar este método para inferência, visto que pode aumentar a precisão
    
    Retorna:
    - k_efetivo: número ideal de segmentos.
    - distâncias: distâncias correspondentes.
    """
    _, distances = hnsw.knn_query(point, k=max_k)
    distances = distances[0]  # Distâncias para os k mais próximos

    # Calcular o coeficiente de variação (CV) para diferentes valores de k
    for k in range(5, max_k + 1):
        subset = distances[:k]
        cv = np.std(subset) / np.mean(subset)
        if cv < cv_threshold:
            return k, distances[:k]

    return max_k, distances  # Retorna o máximo se nenhum CV satisfizer o limite


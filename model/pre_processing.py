import numpy as np
from sklearn.decomposition import PCA
from MetaParams import default_meta_params

'''
    Aqui utilizamos uma única transformação PCA para todas as classes para simplificar o algoritmo e 
    criar uma representação coesa para a comparação na fase de operação. 
    No entanto, poderíamos ter usado um número de componentes diferente para cada classe,
    o que exigiria um ajuste mais detalhado na fase de operação para comparar espaços vetoriais de diferentes dimensões. 

    Além disso, desta maneira conseguimos encapsular todos os segmentos de todas as classes em uma única representação 
    hierárquica de busca, aumentando ainda mais a otimização.     
'''
def reduce_dim_PCA(train_data, meta_params=default_meta_params): 
    threshold = meta_params.variance_perc_threshold
    pca = PCA(n_components=threshold, random_state=2)
    pca.fit(train_data)
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    k = np.argmax(cumulative_var >= threshold) + 1
    return pca, k
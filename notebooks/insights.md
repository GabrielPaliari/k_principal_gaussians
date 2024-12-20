O valor do tamanho do segmento é proporcional à dimensionalidade do problema e a função Qui quadrado: lambda =
$$\text{Scaling Factor} = \sqrt{\chi^2_{n, p}}$$

No cálculo das curvas principais, geralmente se utilizam parâmetros arbitrários, como o tamanho de cada segmento ou a suavidade da curva, ou a distância ao redor da curva que determina que um ponto pertence a ela. Apesar de serem determinadas com relações à variáveis do problema, até agora, pelo que eu li, os valores tendem a assumir uma perspectiva apenas local ou apenas global. Por exemplo, se considero que a área ao redor da curva, ou Outlier Rate, é um número constante para toda a curva, eu desconsidero as diferenças nas variâcias locais de cada segmento. Se, por outro lado, eu considerar apenas o desvio padrão de cada segmento e não olhar para os dados como um todo, posso chegar a mínimos locais, como Verbeek aponta. o desafio deste problema então é conseguir caracterizar os dados por uma curva de forma satisfatória, sem perder o poder de generalização, e fazer isso com uma alta performance, tanto no treino, quanto na fase de operação. 
    
No algoritmo do Verbeek, de K segmentos, o treinamento é feito uma vez, de forma não indutiva. Um dado novo chega, é classificado, mas o cluster não é refeito, pois isso implicaria recalcular todos os segmentos. É preciso que a cada dado novo os segmentos se ajustem de forma indutiva, para que em fase de operação a curva possa se remodelar.
    
Um outro ponto é a ideia de utilizar outras componentes principais para construir um hiper espaço em torno da curva que melhor traduza qual a região daquela classe. 
    
    No caso de múltiplas classes, o tamanho dos segmentos de conexão pode ser considerado. Se o tamanho das conexões for maior que 2 desvios padrões, por exemplo, das conexões existentes, a conexão é removida, e é criada uma nova classe.
    
    - Usar scipy.linalg ao invés de numpy.linalg, pois a do scypi vem obrigatoriamente com LAPLACK e BLAS
    - Acredito que podemos usar o pacote da sci kit learn para criar um modelo \Gaussian Mixture Model\ para aproximar os dados para gaussianas elípticas (não esquecer de usar o processo de Dirichlet antes para diminuir o overfitting). Depois disso, seria possível usar o resultado que é dado como um conjunto de matrizes de covariancia, cada matriz descrevendo uma gaussiana. Podemos então usar o primeiro componente principal como a direção do segmento 
    
    https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_sin.html 
    
    Ler mais sobre covariance estimate and robust covariance estimate LedoitWolf vs OAS vs Empirico. É mais performático e menos vulnerável a erros. Verificar como o **Gaussian Mixture Model** estima a covariância de cada gaussiana
    


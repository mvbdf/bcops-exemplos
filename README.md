# bcops-exemplos

Coleção de exemplos do uso do algoritmo BCOPS, utilizando dados sintéticos e reais. Para cada um dos exemplo temos uma versão utilizando o algoritmo auxiliar Floresta Aleatória e ElasticNet

## Descrição dos casos:

- exemplo1.R: uso do algoritmo com dados gerados contendo duas classes no conjunto de treino e uma classe de outliers adicional no conjunto de teste
- exemplo2.R: uso do algoritmo com dados gerados contendo dez classes no conjunto de treino e uma classe de outliers adicional no conjunto de teste
- exemplo_mnist.R: uso do algoritmo na classificação da base MNIST utilizando os digitos de 0 a 5 no conjunto de treino e incluindo os digitos de 6 a 9 como outliers no conjunto de teste
- exemplo1_ruido.R: avaliação da performance do algoritmo considerando a adição de ruído às classes do conjunto de treino no exemplo 1
- exemplo2_ruido.R: avaliação da performance do algoritmo considerando a adição de ruído às classes do conjunto de treino no exemplo 2
- exemplo_mnist_ruido.R: avaliação da performance do algoritmo considerando a adição de ruído às classes do conjunto de treino no exemplo com a base MNIST

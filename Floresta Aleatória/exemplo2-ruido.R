set.seed(123)

library(bcops)
library(ranger)

## Variáveis relacionadas com o ruído 
space = seq(0, 1, 0.01)
abstention_rate = numeric(101)
evaluations = list()

for (u in 1:101){
  ## Constrói o conjunto de dados
  X_train = matrix(NA, nrow = 5000, ncol = 10)
  y_train = numeric(5000)
  
  X_test = matrix(NA, nrow = 5500, ncol = 10)
  y_test = numeric(5500)
  
  for (i in 1:500){
    ## Dados para o treino
    for (j in 1:10){
      index = i + 500*(j-1)
      X_train[index,] = rnorm(10)
      X_train[index,j] = rnorm(1, 3, 0.5)
      y_train[index] = j
    }
    
    ## Dados para o teste
    for (j in 1:10){
      index = i + 500*(j-1)
      X_test[index,] = rnorm(10)
      X_test[index,j] = rnorm(1, 3, 0.5)
      y_test[index] = j
    }
    
    X_test[i + 5000,] = rnorm(10, 3, 2)
    y_test[i + 5000] = 11
  }
  labels = sort(unique(y_train))
  
  ## Gerando ruído
  y_train_clean = y_train
  y_train = numeric(5000)
  noise_rate = space[u]
  
  for (i in 1:length(y_train)){
    if (rbinom(1, 1, noise_rate))
      y_train[i] = sample(1:10, 1) # corrompe a label aleatóriamente
    else
      y_train[i] = y_train_clean[i]
  }
  
  1-sum(y_train == y_train_clean)/1000 # taxa de ruído
  
  foldid = sample(1:2, length(y_train), replace = TRUE)
  foldid_te = sample(1:2,length(y_test), replace = TRUE)
  xtrain1 = X_train[foldid==1,]
  xtrain2 = X_train[foldid==2,]
  ytrain1 = y_train[foldid==1]
  ytrain2 = y_train[foldid==2]
  xtest1 = X_test[foldid_te ==1,]
  xtest2 = X_test[foldid_te==2,]
  
  bcops = BCOPS(ranger, xtrain1, ytrain1, xtest1, xtrain2, ytrain2, xtest2, 
                labels, formula = T, prediction_only = F)
  prediction.conformal = matrix(NA, ncol = length(labels), nrow = length(y_test))
  prediction.conformal[foldid_te==1,] = bcops[[1]];
  prediction.conformal[foldid_te==2,] = bcops[[2]];
  
  evaluation = evaluate.conformal(prediction.conformal, y_test, labels, 0.05)
  
  ## Conjunto de predição C(x)
  y_pred = list()
  for (i in 1:dim(prediction.conformal)[1]){
    pred = (prediction.conformal[i,] > 0.05)
    y_pred[[i]] = labels[pred]
  }
  
  a = 0 # número de abstenções no conjunto de outliers
  for (i in 5001:5500){
    if (length(y_pred[[i]]) == 0)
      a = a + 1
  }
  
  abstention_rate[u] = a / 500 # taxa de abstenção para outliers
  evaluations[[u]] = evaluation
}

save.image("estadof.RData")

## Evolução da taxa de abstenção
pdf("abstencao.pdf")
plot(abstention_rate, type="b", pch = 18, xlab = "Porcentagem de ruído",
     ylab  = "Taxa de abstenção")
dev.off()

## Evolução da garantia de cobertura
coverage_class = matrix(NA, nrow = length(labels), ncol = 101)

for (i in labels) {
  for (k in 1:101){
    coverage_class[i, k] = evaluations[[k]][i,i]
  }
}

pdf("cobertura.pdf")
plot(coverage_class[1,], type="b", pch = 18, xlab = "Porcentagem de ruído",
     ylab  = "Cobertura", col = "blue")
for (i in 2:10){
  lines(coverage_class[i,], type="b", pch = 18, xlab = "Porcentagem de ruído",
         ylab  = "Cobertura", col = colors()[5*i])
}

dev.off()

average_class_cover = numeric(101)
for (i in 1:101){
  average_class_cover[i] = mean(coverage_class[,i])
}

pdf("cobertura_media.pdf")
plot(average_class_cover, type="b", pch = 18, xlab = "Porcentagem de ruído",
     ylab  = "Cobertura média", col = "black")
dev.off()

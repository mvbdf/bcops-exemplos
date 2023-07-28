set.seed(123)

library(bcops)
library(glmnet)
library(plot.matrix)

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

foldid = sample(1:2, length(y_train), replace = TRUE)
foldid_te = sample(1:2,length(y_test), replace = TRUE)
xtrain1 = X_train[foldid==1,]
xtrain2 = X_train[foldid==2,]
ytrain1 = y_train[foldid==1]
ytrain2 = y_train[foldid==2]
xtest1 = X_test[foldid_te ==1,]
xtest2 = X_test[foldid_te==2,]
labels = sort(unique(y_train))

bcops = BCOPS(cv.glmnet, xtrain1, ytrain1, xtest1, 
              xtrain2, ytrain2, xtest2, labels, formula = FALSE)
prediction.conformal = matrix(NA, ncol = length(labels), nrow = length(y_test))
prediction.conformal[foldid_te==1,] = bcops[[1]];
prediction.conformal[foldid_te==2,] = bcops[[2]];

evaluation = evaluate.conformal(prediction.conformal, y_test, labels, 0.05)

par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(as.matrix(evaluation), digits = 3, col = viridis::viridis,
     ylab = "Classe verdadeira", xlab = "Classe predita",
     main = "")

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
a / 500

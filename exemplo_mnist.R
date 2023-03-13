library(bcops)
library(ranger)
library(plot.matrix)

set.seed(123)

load_image_file <- function(filename) {
  ret = list()
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  ret$n = readBin(f,'integer',n=1,size=4,endian='big')
  nrow = readBin(f,'integer',n=1,size=4,endian='big')
  ncol = readBin(f,'integer',n=1,size=4,endian='big')
  x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
  ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
  close(f)
  ret
}

load_label_file <- function(filename) {
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  n = readBin(f,'integer',n=1,size=4,endian='big')
  y = readBin(f,'integer',n=n,size=1,signed=F)
  close(f)
  y
}

opar <- par(no.readonly = TRUE)      # make a copy of current settings

train <- load_image_file("train-images.idx3-ubyte")
test <- load_image_file("t10k-images.idx3-ubyte")

train$y <- load_label_file("train-labels.idx1-ubyte")
test$y <- load_label_file("t10k-labels.idx1-ubyte")  

par(mfrow = c(5,5))
par(mar=c(0.1,0.1,0.1,0.1))
for (i in 1:25) image(matrix(train$x[i,], 28, 28)[,c(28:1)]) # imagens
matrix(train$y[1:25], 5, 5, byrow = T) # classes

par(opar)

barplot(table(train$y), main = "Distribuição das classes do conjunto de treino original")
sort(unique(train$y))

## Separando o conjunto de treino
mask = train$y %in% c(0, 1, 2, 3, 4, 5)

X_train = matrix(train$x[mask], ncol = 784)
y_train = train$y[mask]

# exemplos do novo treino
par(mfrow = c(5,5))
par(mar=c(0.1,0.1,0.1,0.1))
for (i in 1:25) image(matrix(X_train[i,], 28, 28)[,c(28:1)]) # imagens
matrix(y_train[1:25], 5, 5, byrow = T) # classes

par(opar)

barplot(table(y_train), main = "Distribuição das classes do novo conjunto de treino")
sort(unique(y_train))

X_test = test$x
y_test = test$y

barplot(table(y_test), main = "Distribuição das classes do conjunto de teste")
sort(unique(y_test))

par(mfrow = c(5,5))
par(mar=c(0.1,0.1,0.1,0.1))
for (i in 1:25) image(matrix(test$x[i,], 28, 28)[,c(28:1)]) # imagens
matrix(test$y[1:25], 5, 5, byrow = T) # classes

par(opar)

## Two-fold separation
foldid = sample(1:2, length(y_train), replace = TRUE)
foldid_te = sample(1:2,length(y_test), replace = TRUE)
xtrain1 = X_train[foldid==1,]
xtrain2 = X_train[foldid==2,]
ytrain1 = y_train[foldid==1]
ytrain2 = y_train[foldid==2]
xtest1 = X_test[foldid_te ==1,]
xtest2 = X_test[foldid_te==2,]
labels = sort(unique(y_train))

## Treinamento e predição

bcops = BCOPS(ranger, xtrain1, ytrain1, xtest1, xtrain2, ytrain2, xtest2, 
              labels, formula = T, prediction_only = F)
prediction.conformal = matrix(NA, ncol = length(labels), nrow = length(y_test))
prediction.conformal[foldid_te==1,] = bcops[[1]];
prediction.conformal[foldid_te==2,] = bcops[[2]];

evaluation = evaluate.conformal(prediction.conformal, y_test, labels, 0.05)

par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(as.matrix(evaluation), digits = 3, col = viridis::viridis,
     ylab = "Classe Real", xlab = "Classe predita",
     main = "Porcentagem de observações classificadas")

## Conjunto de predição C(x)
y_pred = list()
for (i in 1:dim(prediction.conformal)[1]){
  pred = (prediction.conformal[i,] > 0.05)
  y_pred[[i]] = labels[pred]
}

## Taxa de abstenção
mask = !(test$y %in% c(0, 1, 2, 3, 4, 5)) # filtra para os outliers

outliers_pred = y_pred[mask]

a = 0 # número de abstenções no conjunto de outliers
for (i in 1:length(outliers_pred)){
  if (length(outliers_pred[[i]]) == 0)
    a = a + 1
}
a / length(outliers_pred)

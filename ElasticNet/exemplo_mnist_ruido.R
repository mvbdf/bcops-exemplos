library(bcops)
library(glmnet)

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

train <- load_image_file("train-images.idx3-ubyte")
test <- load_image_file("t10k-images.idx3-ubyte")

train$y <- load_label_file("train-labels.idx1-ubyte")
test$y <- load_label_file("t10k-labels.idx1-ubyte")  


## Separando o conjunto de treino
mask = train$y %in% c(0, 1, 2, 3, 4, 5)

X_train = matrix(train$x[mask], ncol = 784)
y_train = train$y[mask]

X_test = test$x
y_test = test$y

U = 101
space = seq(0, 1, 0.01)
abstention_rate = numeric(U)
evaluations = list()
y_train_clean = y_train
labels = sort(unique(y_train))

for (u in 1:U){
  
  ## Gerando ruído
  y_train = numeric(length(y_train_clean))
  noise_rate = space[u]
  
  for (i in 1:length(y_train)){
    if (rbinom(1, 1, noise_rate))
      y_train[i] = sample(labels, 1) # corrompe a label aleatóriamente
    else
      y_train[i] = y_train_clean[i]
  }
  
  ## Two-fold separation
  foldid = sample(1:2, length(y_train), replace = TRUE)
  foldid_te = sample(1:2,length(y_test), replace = TRUE)
  xtrain1 = X_train[foldid==1,]
  xtrain2 = X_train[foldid==2,]
  ytrain1 = y_train[foldid==1]
  ytrain2 = y_train[foldid==2]
  xtest1 = X_test[foldid_te ==1,]
  xtest2 = X_test[foldid_te==2,]
  
  ## Treinamento e predição
  
  bcops = BCOPS(cv.glmnet, xtrain1, ytrain1, xtest1, 
                xtrain2, ytrain2, xtest2, labels, formula = FALSE)
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
  
  ## Taxa de abstenção
  mask = !(test$y %in% c(0, 1, 2, 3, 4, 5)) # filtra para os outliers
  
  outliers_pred = y_pred[mask]
  
  a = 0 # número de abstenções no conjunto de outliers
  for (i in 1:length(outliers_pred)){
    if (length(outliers_pred[[i]]) == 0)
      a = a + 1
  }
  abstention_rate[u] = a / length(outliers_pred)
  evaluations[[u]] = evaluation
}

save.image("estadof.RData")

## Evolução da taxa de abstenção
pdf("abstencao.pdf")
plot(abstention_rate, type="b", pch = 18, xlab = "Porcentagem de ruído",
     ylab  = "Taxa de abstenção")
dev.off()

## Evolução da garantia de cobertura
coverage_class = matrix(NA, nrow = length(labels), ncol = U)

for (i in 1:length(labels)) {
  for (k in 1:U){
    coverage_class[i, k] = evaluations[[k]][i,i]
  }
}

pdf("cobertura.pdf")
plot(coverage_class[1,], type="b", pch = 18, xlab = "Porcentagem de ruído",
     ylab  = "Cobertura", col = "blue")
for (i in 2:5){
  lines(coverage_class[i,], type="b", pch = 18, xlab = "Porcentagem de ruído",
        ylab  = "Cobertura", col = colors()[5*i])
}
dev.off()

average_class_cover = numeric(U)
for (i in 1:U){
  average_class_cover[i] = mean(coverage_class[,i])
}

pdf("cobertura_media.pdf")
plot(average_class_cover, type="b", pch = 18, xlab = "Porcentagem de ruído",
     ylab  = "Cobertura média", col = "blue")
dev.off()

library(bcops)
library(ranger)
library(plot.matrix)

set.seed(123)

X_train = matrix(NA, nrow = 1000, ncol = 10)
y_train = numeric(1000)

X_test = matrix(NA, nrow = 1500, ncol = 10)
y_test = numeric(1500)

for (i in 1:500){
  ## Dados para o treino
  # Classe 1
  X_train[i,] = rnorm(10)
  y_train[i] = 1
  
  # Classe 2
  X_train[i + 500,] = c(rnorm(1, mean = 3, sd = 0.5), rnorm(9))
  y_train[i + 500] = 2
  
  ## Dados para o teste
  # Classe 1
  X_test[i,] = rnorm(10)
  y_test[i] = 1
  
  # Classe 2
  X_test[i + 500,] = c(rnorm(1, mean = 3, sd = 0.5), rnorm(9))
  y_test[i + 500] = 2
  
  # Classe 3 (outliers)
  X_test[i + 1000,] = c(rnorm(1), rnorm(1, mean = 3, sd = 1), rnorm(8))
  y_test[i + 1000] = 3
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

bcops = BCOPS(ranger, xtrain1, ytrain1, xtest1, xtrain2, ytrain2, xtest2, 
              labels, formula = T, prediction_only = F)
prediction.conformal = matrix(NA, ncol = length(labels), nrow = length(y_test))
prediction.conformal[foldid_te==1,] = bcops[[1]];
prediction.conformal[foldid_te==2,] = bcops[[2]];

alpha = 1-0.95

y_pred = numeric(dim(prediction.conformal)[1])
for (i in 1:dim(prediction.conformal)[1]){
  pred = (prediction.conformal[i,] > alpha)
  
  if (pred[1] == F & pred[2] == F)
    y_pred[i] = 3 # classe de outliers
  else if (pred[1] == T & pred[2] == T)
    y_pred[i] = 4 # ambas as classes
  else
    y_pred[i] = labels[pred]
}

colors = c("black", "blue", "red", "green")
plot(X_train[,1], X_train[,2], col = colors[factor(y_train)],
     main = "", xlab = expression(x[1]), ylab = expression(x[2]))
legend("topleft", fill = c("black", "blue"),
       legend = c("1", "2"), cex = 0.7)

plot(X_test[,1], X_test[,2], col = colors[factor(y_test)],
     main = "", xlab = expression(x[1]), ylab = expression(x[2]))
legend("topright", fill = c("black", "blue", "red"),
       legend = c("1", "2", "R"), cex = 0.7)

plot(X_test[,1], X_test[,2], col = colors[factor(y_pred)],
     main = "", xlab = expression(x[1]), ylab = expression(x[2]))
legend("topright", fill = c("black", "blue", "red", "green"),
       legend = c("1", "2", "R", "B"), cex = 0.7)

evaluation = evaluate.conformal(prediction.conformal, y_test, labels, alpha)

par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(as.matrix(evaluation), digits = 3, col = viridis::viridis,
     ylab = "Classe verdadeira", xlab = "Classe predita",
     main = "")

mean(y_pred[1000:1500] == 3) # taxa de abstenção para outliers

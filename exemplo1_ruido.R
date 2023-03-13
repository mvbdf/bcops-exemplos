set.seed(123)

library(bcops)
library(ranger)
library(plot.matrix)
library(zoo) # moving averages

space = seq(0, 1, 0.01)
colors = c("black", "blue", "red", "green")

# abstention_rate =  numeric(101)
abstention_rate = matrix(NA, 101, 101)
evaluations = list()

save_xtrain = list()
save_xtest = list()
save_ytrain = list()
save_ytrain_clean = list()
save_ytest = list()

for (k in 1:101){
  for (j in 1:101){
  
    noise_rate = space[j]
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
    
    ## Gerando ruído
    y_train_clean = y_train
    y_train = numeric(1000)
    
    for (i in 1:length(y_train)){
      if (rbinom(1, 1, noise_rate)) # inverte a classe da observação
        y_train[i] = if (y_train_clean[i] == 1) 2 else 1
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
    
    evaluation = evaluate.conformal(prediction.conformal, y_test, labels, alpha)
    
    abstention_rate[k, j] = length(y_pred[y_pred == 3]) / 500 # taxa de abstenção para outliers
    # evaluations[[j]] = evaluation
    
    save_xtrain[[j]]       = X_train
    save_xtest[[j]]        = X_test
    save_ytrain[[j]]       = y_train
    save_ytrain_clean[[j]] = y_train_clean
    save_ytest[[j]]        = y_test
  }
}

## Evolução da taxa de abstenção
plot(abstention_rate, type="b", pch = 18, xlab = "Porcentagem de ruído",
     ylab  = "Taxa de abstenção")
lines(rollmean(abstention_rate, 10), col = "red")

## Evolução da média da taxa de abstenção
mean_abstention_rate = numeric(101)
for (i in 1:101){
  mean_abstention_rate[i] = mean(abstention_rate[,i])
}
plot(mean_abstention_rate, type="b", pch = 18, xlab = "Porcentagem de ruído",
     ylab  = "Taxa de abstenção média")
lines(rollmean(mean_abstention_rate, 10), col = "red")


## Evolução da garantia de cobertura
coverage_class1 = numeric(101)
coverage_class2 = numeric(101)

for (i in 1:101) {
  coverage_class1[i] = evaluations[[i]][1,1]
  coverage_class2[i] = evaluations[[i]][2,2]
}

plot(coverage_class1, type="b", pch = 18, xlab = "Porcentagem de ruído",
     ylab  = "Cobertura", col = "blue")
points(coverage_class2, type="b", pch = 18, xlab = "Porcentagem de ruído",
     ylab  = "Cobertura")
legend("bottomleft", legend = c("Classe 1", "Classe 2"), fill = c("blue", "black"))


## Máximo e mínimo da taxa de abstenção
which(abstention_rate == max(abstention_rate)) # ponto de maior abstenção
which(abstention_rate == min(abstention_rate)) # ponto de menor abstenção


##### Plot de uma iteração específica #####
{
  i = 8
  X_train = save_xtrain[[i]]
  y_train = save_ytrain[[i]]
  X_test = save_xtest[[i]]
  y_test = save_ytest[[i]]
  y_test_clean = save_ytrain_clean[[i]]
  
  plot(X_train[,1], X_train[,2], col = colors[factor(y_train)],
       main = paste("Dados de treinamento", i), xlab = "X1", ylab = "X2")
  legend("topleft", fill = c("black", "blue"),
         legend = c("1", "2"), cex = 0.7)
  
  plot(X_test[,1], X_test[,2], col = colors[factor(y_test)],
       main = paste("Dados de teste", i), xlab = "X1", ylab = "X2")
  legend("topright", fill = c("black", "blue", "red"),
         legend = c("1", "2", "R"), cex = 0.7)
  
  plot(X_test[,1], X_test[,2], col = colors[factor(y_pred)],
       main = paste("Classe predita", i), xlab = "X1", ylab = "X2")
  legend("topright", fill = c("black", "blue", "red", "green"),
         legend = c("1", "2", "R", "B"), cex = 0.7)
  
  par(mar=c(5.1, 4.1, 4.1, 4.1))
  plot(as.matrix(evaluation), digits = 3, col = viridis::viridis,
       ylab = "Classe Real", xlab = "Classe predita",
       main = paste("Porcentagem de observações classificadas", i))
  abstention_rate[i]
}

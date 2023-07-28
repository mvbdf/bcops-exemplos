set.seed(123)

library(bcops)
library(ranger)

space = seq(0, 1, 0.01)

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
    labels = sort(unique(y_train))
    
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
    
    abstention_rate[k, j] = mean(y_pred[1000:1500] == 3) # taxa de abstenção para outliers
    evaluations[[j]] = evaluation
    
    save_xtrain[[j]]       = X_train
    save_xtest[[j]]        = X_test
    save_ytrain[[j]]       = y_train
    save_ytrain_clean[[j]] = y_train_clean
    save_ytest[[j]]        = y_test
  }
}

save.image("estadof.RData")

## Evolução da média da taxa de abstenção
mean_abstention_rate = numeric(101)
for (i in 1:101){
  mean_abstention_rate[i] = mean(abstention_rate[,i])
}

pdf("abstencao.pdf") 
plot(mean_abstention_rate, type="b", pch = 18, xlab = "Porcentagem de ruído",
     ylab  = "Taxa de abstenção média")
dev.off()

## Evolução da garantia de cobertura

coverage_class1 = numeric(101)
coverage_class2 = numeric(101)

for (i in 1:101) {
  coverage_class1[i] = evaluations[[i]][1,1]
  coverage_class2[i] = evaluations[[i]][2,2]
}

pdf("cobertura.pdf")
plot(coverage_class1, type="b", pch = 18, xlab = "Porcentagem de ruído",
     ylab  = "Cobertura", col = "blue")
points(coverage_class2, type="b", pch = 18, xlab = "Porcentagem de ruído",
       ylab  = "Cobertura")
legend("bottomleft", legend = c("Classe 1", "Classe 2"), fill = c("blue", "black"))
dev.off()

library(caret)
library(dplyr)
library(pROC)
library(tidyr)
library(imbalance)

learn_model <-function(dataset, ctrl,message){
  knn.fit <- train(Class ~ ., data = dataset, method = "knn", 
                   trControl = ctrl, preProcess = c("center","scale"), metric="ROC", 
                   tuneGrid = expand.grid(k = c(1,3,5,7,9,11)))
  knn.pred <- predict(knn.fit,newdata = dataset)
  #Get the confusion matrix to see accuracy value and other parameter values
  knn.cm <- confusionMatrix(knn.pred, dataset$Class,positive = "positive")
  knn.probs <- predict(knn.fit,newdata = dataset, type="prob")
  knn.roc <- roc(dataset$Class,knn.probs[,"positive"],color="green")
  return(knn.fit)
}

test_model <-function(dataset, knn.fit,message){
  knn.pred <- predict(knn.fit,newdata = dataset)
  #Get the confusion matrix to see accuracy value and other parameter values
  knn.cm <- confusionMatrix(knn.pred, dataset$Class,positive = "positive")
  print(knn.cm)
  knn.probs <- predict(knn.fit,newdata = dataset, type="prob")
  knn.roc <- roc(dataset$Class,knn.probs[,"positive"])
  #print(knn.roc)
  plot(knn.roc, type="S", print.thres= 0.5,main=c("ROC Test",message),col="blue")
  #print(paste0("AUC Test ",message,auc(knn.roc)))
  return(knn.cm)
}

# 3. Análisis del efecto del desbalanceo en problemas de clasificación

#En primer lugar trabajaremos con los datos subclus.txt

#load dataset subclus
dataset <- read.table("subclus.txt", sep=",")
#dataset <- read.table("circle.txt", sep=",")
colnames(dataset) <- c("Att1", "Att2", "Class")
summary(dataset)

# visualize the data distribution
plot(dataset$Att1, dataset$Att2)
points(dataset[dataset$Class=="negative",1],dataset[dataset$Class=="negative",2],col="red")
points(dataset[dataset$Class=="positive",1],dataset[dataset$Class=="positive",2],col="blue")  

#Additional:
data(glass0)
#dataset <- glass0

imbalanceRatio(dataset) # Me devuelve 0.2 entonces representa que nos encontramos delante de un dataset desbalanceado.

#Create Data Partition
set.seed(42)
dataset$Class <- relevel(dataset$Class,"positive")
index <- createDataPartition(dataset$Class, p = 0.7, list = FALSE)
train_data <- dataset[index, ]
test_data  <- dataset[-index, ]

#Execute model ("raw" data)
ctrl <- trainControl(method="repeatedcv",number=5,repeats = 3,
                     classProbs=TRUE,summaryFunction = twoClassSummary)

model.subclus.raw <- learn_model(train_data,ctrl,"RAW ")

## Aplicamos el modelo con Random Undersampling
ctrl <- trainControl(method="repeatedcv",number=5,repeats = 3,
                     classProbs=TRUE,summaryFunction = twoClassSummary, sampling = "down") #RUS
model.subclus.us <- learn_model(train_data,ctrl,"US ")

## Aplicamos el modelo con Random Oversampling
ctrl <- trainControl(method="repeatedcv",number=5,repeats = 3,
                     classProbs=TRUE,summaryFunction = twoClassSummary, sampling = "up") #ROS
model.subclus.os <- learn_model(train_data,ctrl,"OS ")

## Aplicamos el modelo con Synthetic Minority Oversampling Technique 
ctrl <- trainControl(method="repeatedcv",number=5,repeats = 3,
                     classProbs=TRUE,summaryFunction = twoClassSummary, sampling = "smote") #SMOTE
model.subclus.smt <- learn_model(train_data,ctrl,"SMT ")

#plot(model,main="Grid Search RAW")
#print(model.raw)
cm.subclus.raw <- test_model(test_data,model.subclus.raw,"RAW ")

cm.subclus.us <- test_model(test_data,model.subclus.us,"US ") #undersampling

cm.subclus.os <- test_model(test_data,model.subclus.os,"OS ") #oversampling

cm.subclus.smt <- test_model(test_data,model.subclus.smt,"SMT ")

#Execute model ("preprocessed" data)
#Undersampling
#ctrl <- trainControl(method="repeatedcv",number=5,repeats = 3, classProbs=TRUE,summaryFunction = twoClassSummary,sampling = "down")

#Check model's behavior
models <- list(raw = model.subclus.raw,
               us = model.subclus.us,
               os = model.subclus.os,
               smt = model.subclus.smt)

resampling <- resamples(models)
bwplot(resampling)

comparison <- data.frame(model = names(models),
                         Sensitivity = rep(NA, length(models)),
                         Specificity = rep(NA, length(models)),
                         Precision = rep(NA, length(models)),
                         Recall = rep(NA, length(models)),
                         F1 = rep(NA, length(models)))

for (name in names(models)) {
  cm_model <- get(paste0("cm.", name))
  
  comparison[comparison$model == name, ] <- filter(comparison, model == name) %>%
    mutate(Sensitivity = cm_model$byClass["Sensitivity"],
           Specificity = cm_model$byClass["Specificity"],
           Precision = cm_model$byClass["Precision"],
           Recall = cm_model$byClass["Recall"],
           F1 = cm_model$byClass["F1"])
}

comparison %>%
  gather(x, y, Sensitivity:F1) %>%
  ggplot(aes(x = x, y = y, color = model)) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 3)

## como podemos observar en el gráfico, los modelos en precision y specificitydestaca el modelo "raw"  pero a cambio
## posee valores bastante malos es recall y sensivity. En F1 se encuentra en un punto "medio" ya que dicho parámetro se 
## se calcula a partir de la precicison y recall, que posee valores alto y bajo respectivamente.
## En recall y sensivity destacan los moelso us, os y smt (en este orden)

#load dataset circle
dataset <- read.table("circle.txt", sep=",")
colnames(dataset) <- c("Att1", "Att2", "Class")
summary(dataset)

# visualize the data distribution
plot(dataset$Att1, dataset$Att2)
points(dataset[dataset$Class=="negative",1],dataset[dataset$Class=="negative",2],col="red")
points(dataset[dataset$Class=="positive",1],dataset[dataset$Class=="positive",2],col="blue")  

#Additional:
data(glass0)
#dataset <- glass0

imbalanceRatio(dataset)
# En este dataset poseemos un imablance ratio peor que en datatset subclus


#Create Data Partition
set.seed(42)
dataset$Class <- relevel(dataset$Class,"positive")
index <- createDataPartition(dataset$Class, p = 0.7, list = FALSE)
train_data <- dataset[index, ]
test_data  <- dataset[-index, ]

#Execute model ("raw" data)
ctrl <- trainControl(method="repeatedcv",number=5,repeats = 3, classProbs=TRUE,summaryFunction = twoClassSummary)

model.circle.raw <- learn_model(train_data,ctrl,"RAW ")
##
ctrl <- trainControl(method="repeatedcv",number=5,repeats = 3, classProbs=TRUE,summaryFunction = twoClassSummary, sampling = "down") #RUS
model.circle.us <- learn_model(train_data,ctrl,"US ")

ctrl <- trainControl(method="repeatedcv",number=5,repeats = 3, classProbs=TRUE,summaryFunction = twoClassSummary, sampling = "up") #ROS
model.circle.os <- learn_model(train_data,ctrl,"OS ")

ctrl <- trainControl(method="repeatedcv",number=5,repeats = 3, classProbs=TRUE,summaryFunction = twoClassSummary, sampling = "smote") #SMOTE
model.circle.smt <- learn_model(train_data,ctrl,"SMT ")

#plot(model,main="Grid Search RAW")
#print(model.raw)
cm.circle.raw <- test_model(test_data,model.circle.raw,"RAW ")

cm.circle.us <- test_model(test_data,model.circle.us,"US ") #undersampling

cm.circle.os <- test_model(test_data,model.circle.os,"OS ") #oversampling

cm.circle.smt <- test_model(test_data,model.circle.smt,"SMT ")

#Execute model ("preprocessed" data)
#Undersampling
ctrl <- trainControl(method="repeatedcv",number=5,repeats = 3,
                     classProbs=TRUE,summaryFunction = twoClassSummary,sampling = "down")

#Check model's behavior
models <- list(raw = model.circle.raw,
               us = model.circle.us,
               os = model.circle.os,
               smt = model.circle.smt)

resampling <- resamples(models)
bwplot(resampling)

comparison <- data.frame(model = names(models),
                         Sensitivity = rep(NA, length(models)),
                         Specificity = rep(NA, length(models)),
                         Precision = rep(NA, length(models)),
                         Recall = rep(NA, length(models)),
                         F1 = rep(NA, length(models)))

for (name in names(models)) {
  cm_model <- get(paste0("cm.", name))
  
  comparison[comparison$model == name, ] <- filter(comparison, model == name) %>%
    mutate(Sensitivity = cm_model$byClass["Sensitivity"],
           Specificity = cm_model$byClass["Specificity"],
           Precision = cm_model$byClass["Precision"],
           Recall = cm_model$byClass["Recall"],
           F1 = cm_model$byClass["F1"])
}

comparison %>%
  gather(x, y, Sensitivity:F1) %>%
  ggplot(aes(x = x, y = y, color = model)) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 3)

## En este caso observamos que el modelo donde mayoritariamente posee mejores valores entre las distintas métricas es 
## es el modelo raw. Cabe destacar la poca precisión  y por tanto el valor bajo en F1 del modelo us. Finalmente respecto al resto de modelos,
## poseen valores altos principalemte en Recall y sensitivity junto con valores medio-altos en precisión F1 y specifity.

# 4. Paquete imbalance y combinación de técnicas

#load dataset
dataset <- read.table("subclus.txt", sep=",")
colnames(dataset) <- c("Att1", "Att2", "Class")
summary(dataset)

# visualize the data distribution
plot(dataset$Att1, dataset$Att2)
points(dataset[dataset$Class=="negative",1],dataset[dataset$Class=="negative",2],col="red")
points(dataset[dataset$Class=="positive",1],dataset[dataset$Class=="positive",2],col="blue")  

#Additional:
data(glass0)
#dataset <- glass0

dataset <- oversample(dataset = dataset, method = "RWO", ratio = 1)
#  Mediante la técnica oversample  obtenemos más datos de la clase minoritaria de las que originalmente había, dejando intacta la cantidad de elementos de la clase mayoritaria.

imbalanceRatio(dataset)

# visualize the data distribution
plot(dataset$Att1, dataset$Att2)
points(dataset[dataset$Class=="negative",1],dataset[dataset$Class=="negative",2],col="red")
points(dataset[dataset$Class=="positive",1],dataset[dataset$Class=="positive",2],col="blue")  

## Si aplicamos oversample con el método SMOTE y ratio 1 poseemos un imbalance ratio de 1, que equivale que poseemos un un dataset bien balanceado

#Create Data Partition
set.seed(42)
dataset$Class <- relevel(dataset$Class,"positive")
index <- createDataPartition(dataset$Class, p = 0.7, list = FALSE)
train_data <- dataset[index, ]
test_data  <- dataset[-index, ]

#Execute model ("raw" data)
ctrl <- trainControl(method="repeatedcv",number=5,repeats = 3,
                     classProbs=TRUE,summaryFunction = twoClassSummary)

model.subclus.rwo.raw <- learn_model(train_data,ctrl,"RAW ")
cm.subclus.rwo.raw <- test_model(test_data,model.subclus.rwo.raw,"RAW ")

## A continuación comparamos los métodos utilizados anteriormente respecto al nuevo método de oversample utilizado (raw)
#Check model's behavior
models <- list(raw = model.subclus.raw,
               us = model.subclus.us,
               os = model.subclus.os,
               smt = model.subclus.smt,
               rwo = model.subclus.rwo.raw)

resampling <- resamples(models)
bwplot(resampling)

comparison <- data.frame(model = names(models),
                         Sensitivity = rep(NA, length(models)),
                         Specificity = rep(NA, length(models)),
                         Precision = rep(NA, length(models)),
                         Recall = rep(NA, length(models)),
                         F1 = rep(NA, length(models)))

for (name in names(models)) {
  cm_model <- get(paste0("cm.", name))
  
  comparison[comparison$model == name, ] <- filter(comparison, model == name) %>%
    mutate(Sensitivity = cm_model$byClass["Sensitivity"],
           Specificity = cm_model$byClass["Specificity"],
           Precision = cm_model$byClass["Precision"],
           Recall = cm_model$byClass["Recall"],
           F1 = cm_model$byClass["F1"])
}

comparison %>%
  gather(x, y, Sensitivity:F1) %>%
  ggplot(aes(x = x, y = y, color = model)) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 3)

## El metodo rwo aporta valores bastante malos respecto al resto de métodos. Excepto en Specificity, se encuentra en peor lugar en el resto de parámetros.

#load dataset circle
dataset <- read.table("circle.txt", sep=",")
colnames(dataset) <- c("Att1", "Att2", "Class")
summary(dataset)

# visualize the data distribution
plot(dataset$Att1, dataset$Att2)
points(dataset[dataset$Class=="negative",1],dataset[dataset$Class=="negative",2],col="red")
points(dataset[dataset$Class=="positive",1],dataset[dataset$Class=="positive",2],col="blue")

dataset <- oversample(dataset = dataset, method = "RWO", ratio = 1)

imbalanceRatio(dataset)

#Create Data Partition
set.seed(42)
dataset$Class <- relevel(dataset$Class,"positive")
index <- createDataPartition(dataset$Class, p = 0.7, list = FALSE)
train_data <- dataset[index, ]
test_data  <- dataset[-index, ]

#Execute model ("raw" data)
ctrl <- trainControl(method="repeatedcv",number=5,repeats = 3,
                     classProbs=TRUE,summaryFunction = twoClassSummary)

model.circle.rwo.raw <- learn_model(train_data,ctrl,"RAW ")
cm.circle.rwo.raw <- test_model(test_data,model.circle.rwo.raw,"RAW ")

#Undersampling
ctrl <- trainControl(method="repeatedcv",number=5,repeats = 3,
                     classProbs=TRUE,summaryFunction = twoClassSummary,sampling = "down")

#Check model's behavior
models <- list(raw = model.raw,
               us = model.us,
               os = model.os,
               smt = model.smt,
               over = model.circle.rwo.raw)

resampling <- resamples(models)
bwplot(resampling)

comparison <- data.frame(model = names(models),
                         Sensitivity = rep(NA, length(models)),
                         Specificity = rep(NA, length(models)),
                         Precision = rep(NA, length(models)),
                         Recall = rep(NA, length(models)),
                         F1 = rep(NA, length(models)))

for (name in names(models)) {
  cm_model <- get(paste0("cm.", name))
  
  comparison[comparison$model == name, ] <- filter(comparison, model == name) %>%
    mutate(Sensitivity = cm_model$byClass["Sensitivity"],
           Specificity = cm_model$byClass["Specificity"],
           Precision = cm_model$byClass["Precision"],
           Recall = cm_model$byClass["Recall"],
           F1 = cm_model$byClass["F1"])
}

comparison %>%
  gather(x, y, Sensitivity:F1) %>%
  ggplot(aes(x = x, y = y, color = model)) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 3)

# En este datataset rwo posee mejores resultados respecto al dataset subclus. rwo posee un valor alto en precisión y recall  en cambio en sensitivity sigue siendo bastante malo.

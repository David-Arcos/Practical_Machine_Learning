### LIBRARIES ###
library(lattice)
library(ggplot2)
library(caret)
library(kernlab)
library(rattle)
library(corrplot)
library(randomForest)
set.seed(1234)

### READING DATA ###
list.files()
traincsv <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
testcsv <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))

### CLEANING DATA ###

# 1. Remove NA 
traincsv_1 <- traincsv[,colMeans(is.na(traincsv)) == 0] 
str(traincsv_1)

# 2. Remove metadata
traincsv_1 <- traincsv_1[,-c(1:7)] 

# 3ero. Removing near zero variance variables.
nvz <- nearZeroVar(traincsv_1, saveMetrics = T)
nvz

### DATA PARTITIONING ###
# Now that we have finished removing the unnecessary variables, we can now split the training set
# into a validation and sub training set. The testing set “testcsv” will be left alone, 
# and used for the final quiz test cases.
inTrain <- createDataPartition(y=traincsv_1$classe, p=0.7, list=F)
train <- traincsv_1[inTrain,]
valid <- traincsv_1[-inTrain,]

### MODELLING
# Here we will test a few popular models including: Decision Trees, Random Forest, 
# Gradient Boosted Trees, and SVM. This is probably more than we will need to test, but just out 
# of curiosity and good practice we will run them for comparison

control <- trainControl(method="cv", number=3, verboseIter=F)
?trainControl

### DECISION TREE ###
?train
mod_trees <- train(classe~., data=train, method="rpart", 
                   trControl = control, tuneLength = 3)

fancyRpartPlot(mod_trees$finalModel)
?fancyRpartPlot
pred_trees <- predict(mod_trees, valid)
cmtrees <- confusionMatrix(pred_trees, factor(valid$classe))
cmtrees
cmtrees$overall[1]

### RANDOM FOREST ###
mod_rf <- train(classe~., data=train, method="rf", 
                trControl = control)#, tuneLength = 5)

pred_rf <- predict(mod_rf, valid)
cmrf <- confusionMatrix(pred_rf, factor(valid$classe))
cmrf
cmrf$overall[1]

### GRADIENTE BOOST TREES ###
mod_gbm <- train(classe~., data=train, method="gbm", 
                 trControl = control, tuneLength = 5, verbose = F)

pred_gbm <- predict(mod_gbm, valid)
cmgbm <- confusionMatrix(pred_gbm, factor(valid$classe))
cmgbm
cmgbm$overall[1]

### SUPPORT VECTOR MACHINE ###
mod_svm <- train(classe~., data=train, method="svmLinear", 
                 trControl = control, tuneLength = 5, verbose = F)

pred_svm <- predict(mod_svm, valid)
cmsvm <- confusionMatrix(pred_svm, factor(valid$classe))
cmsvm
cmsvm$overall[1]

### RESULTS ###
overall_accuracy <- data.frame(Decision_tree = cmtrees$overall[1],
                               Random_forest = cmrf$overall[1],
                               Gradient_boost = cmgbm$overall[1],
                               SVM = cmsvm$overall[1])

pred <- predict(mod_rf, testcsv)
print(pred)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(pred)

### APPENDIX ###
corrPlot <- cor(train[, -length(names(train))])
corrplot(corrPlot,
         order = "FPC", #original/hclust/AOE/FPC 
         method = "shade",
         type = "lower", 
         tl.col = "black", tl.srt = 35, 
         outline = T,tl.cex = 0.6,
         addgrid.col = "black")
plot(mod_trees)
plot(mod_rf)


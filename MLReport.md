# Introduction

This is the final project for Coursera’s Practical Machine Learning
course, as part of the Data Science Specialization offered by John
Hopkins University.

Using devices such as Jawbone Up, Nike FuelBand, and Fitbitwe, in this
project will use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants to predict the manner in which they did the
exercise using as predictor the ***classe*** variable. For this purpose,
we will run **4** different models using k-folds cross validation and
compare their overall accuracy by using a validation set randomly
selected from the training data. Based on the accuracy, we use the best
model to predict 20 cases using the test dataset.

More information is available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

# Data Sources

The training data for this project is available here:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data is available here:  
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project comes from this original source:
<http://groupware.les.inf.puc-rio.br/har>. If you use the document you
create for this class for any purpose please cite them as they have been
very generous in allowing their data to be used for this kind of
assignment.

# Loading Data and Libraries

``` r
library(lattice)
library(ggplot2)
library(caret)
library(kernlab)
library(rattle)
library(corrplot)
set.seed(1234)
```

``` r
traincsv <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
dim(traincsv)
```

    ## [1] 19622   160

``` r
testcsv <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
dim(testcsv)
```

    ## [1]  20 160

# Data Cleaning

For cleaning the training dataset, **first** we will remove the
variables which consist mostly of NA (with this, we will remove about
100 variables which contain more than 90% of NAs as observations).

``` r
traincsv <- traincsv[,colMeans(is.na(traincsv)) == 0] 
# Keep those variables that doesn't have any NA
```

**Second**, we will remove metadata variables, which happen to be the
first **7** variables.

``` r
traincsv <- traincsv[,-c(1:7)]
```

**Third**, we will remove those variable that don’t have much
variability, for instance, zero or near-zero variance, by using the
`nearZeroVar` command. As we can see, each of the 53 variables left in
the training dataset contribute with some variability.

``` r
nvz <- nearZeroVar(traincsv, saveMetrics = T)
```

# Data Partitioning

Now that we have removed unnecessary variables, we can proceed now by
spliting the training dataset into a validation and sub training set
with a 70/30 proportion. The testing set will be left alone, and used
for the final testing.

``` r
inTrain <- createDataPartition(y=traincsv$classe, p=0.7, list=F)
train <- traincsv[inTrain,]
valid <- traincsv[-inTrain,]
dim(train); dim(valid)
```

    ## [1] 13737    53

    ## [1] 5885   53

Now, we can test 4 different types of models including: **Decision
Trees**, **Random forest**, **Gradient Boosted Trees** and **Support
Vector Machine**. For this, we set up the training controls so that we
can use a 3-fold cross-validation.

``` r
control <- trainControl(method="cv", number=3, verboseIter=F)
```

# Testing models

## Decision trees

``` r
mod_trees <- train(classe~., data=train, method="rpart", 
                   trControl = control, tuneLength = 5)
pred_trees <- predict(mod_trees, valid)
cmtrees <- confusionMatrix(pred_trees, factor(valid$classe))
cmtrees
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1519  473  484  451  156
    ##          B   28  355   45   10  130
    ##          C   83  117  423  131  131
    ##          D   40  194   74  372  176
    ##          E    4    0    0    0  489
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.5366          
    ##                  95% CI : (0.5238, 0.5494)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3957          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9074  0.31168  0.41228  0.38589  0.45194
    ## Specificity            0.6286  0.95512  0.90492  0.90165  0.99917
    ## Pos Pred Value         0.4927  0.62500  0.47797  0.43458  0.99189
    ## Neg Pred Value         0.9447  0.85255  0.87940  0.88228  0.89002
    ## Prevalence             0.2845  0.19354  0.17434  0.16381  0.18386
    ## Detection Rate         0.2581  0.06032  0.07188  0.06321  0.08309
    ## Detection Prevalence   0.5239  0.09652  0.15038  0.14545  0.08377
    ## Balanced Accuracy      0.7680  0.63340  0.65860  0.64377  0.72555

## Random forest

``` r
mod_rf <- train(classe~., data=train, method="rf", 
                trControl = control, tuneLength = 5)
pred_rf <- predict(mod_rf, valid)
cmrf <- confusionMatrix(pred_rf, factor(valid$classe))
cmrf
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    4    0    0    0
    ##          B    1 1132    8    0    0
    ##          C    0    3 1016    5    1
    ##          D    0    0    2  958    0
    ##          E    0    0    0    1 1081
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9958          
    ##                  95% CI : (0.9937, 0.9972)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9946          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9939   0.9903   0.9938   0.9991
    ## Specificity            0.9991   0.9981   0.9981   0.9996   0.9998
    ## Pos Pred Value         0.9976   0.9921   0.9912   0.9979   0.9991
    ## Neg Pred Value         0.9998   0.9985   0.9979   0.9988   0.9998
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2843   0.1924   0.1726   0.1628   0.1837
    ## Detection Prevalence   0.2850   0.1939   0.1742   0.1631   0.1839
    ## Balanced Accuracy      0.9992   0.9960   0.9942   0.9967   0.9994

## Gradient boosted trees

``` r
mod_gbm <- train(classe~., data=train, method="gbm", 
                 trControl = control, tuneLength = 5, verbose = F)
pred_gbm <- predict(mod_gbm, valid)
cmgbm <- confusionMatrix(pred_gbm, factor(valid$classe))
cmgbm
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1671    5    0    0    0
    ##          B    1 1128   15    0    0
    ##          C    2    6 1007    8    4
    ##          D    0    0    4  953    1
    ##          E    0    0    0    3 1077
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9917         
    ##                  95% CI : (0.989, 0.9938)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9895         
    ##                                          
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9982   0.9903   0.9815   0.9886   0.9954
    ## Specificity            0.9988   0.9966   0.9959   0.9990   0.9994
    ## Pos Pred Value         0.9970   0.9860   0.9805   0.9948   0.9972
    ## Neg Pred Value         0.9993   0.9977   0.9961   0.9978   0.9990
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2839   0.1917   0.1711   0.1619   0.1830
    ## Detection Prevalence   0.2848   0.1944   0.1745   0.1628   0.1835
    ## Balanced Accuracy      0.9985   0.9935   0.9887   0.9938   0.9974

## Support vector machine

``` r
mod_svm <- train(classe~., data=train, method="svmLinear", 
                 trControl = control, tuneLength = 5, verbose = F)

pred_svm <- predict(mod_svm, valid)
cmsvm <- confusionMatrix(pred_svm, factor(valid$classe))
cmsvm
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1537  154   79   69   50
    ##          B   29  806   90   46  152
    ##          C   40   81  797  114   69
    ##          D   61   22   32  697   50
    ##          E    7   76   28   38  761
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7813          
    ##                  95% CI : (0.7705, 0.7918)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.722           
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9182   0.7076   0.7768   0.7230   0.7033
    ## Specificity            0.9164   0.9332   0.9374   0.9665   0.9690
    ## Pos Pred Value         0.8137   0.7177   0.7239   0.8086   0.8363
    ## Neg Pred Value         0.9657   0.9301   0.9521   0.9468   0.9355
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2612   0.1370   0.1354   0.1184   0.1293
    ## Detection Prevalence   0.3210   0.1908   0.1871   0.1465   0.1546
    ## Balanced Accuracy      0.9173   0.8204   0.8571   0.8447   0.8362

# Results

``` r
models <- c("Tree", "RF", "GBM", "SVM")
# Accuracy
accuracy <- round(c(cmtrees$overall[1], cmrf$overall[1], cmgbm$overall[1], cmsvm$overall[1]),3) 
# Out-of-sample error
oos_error <- 1 - accuracy
data.frame(accuracy = accuracy, oos_error = oos_error, row.names = models)
```

    ##      accuracy oos_error
    ## Tree    0.537     0.463
    ## RF      0.996     0.004
    ## GBM     0.992     0.008
    ## SVM     0.781     0.219

The best model for predict this dataset is the **Random Forest** model
with a 99.6% accuracy rate and a 0.4% out-of-sample error rate, followed
by the **Gradient Boost Model** with a 99.1% accuracy rate and a 0.9%
out-of-sample rate.

# Predictions on the Test dataset

Finally, we can use the **Random Forest** model to predict the 5 leves
on the *“classe”* variable in the test set for 20 different
observations.

``` r
pred <- predict(mod_rf, testcsv)
print(pred)
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

# Appendix

### Appendix 1. Correlation plot between the 53 variables of the training dataset

![](Machine-Learning_files/figure-markdown_github/unnamed-chunk-15-1.png)

### Appendix 2. Decision tree plot

![](Machine-Learning_files/figure-markdown_github/unnamed-chunk-16-1.png)

### Appendix 3. Random Forest plot

![](Machine-Learning_files/figure-markdown_github/unnamed-chunk-17-1.png)

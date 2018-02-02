# Loading libraries
library(kernlab)
library(readr)
library(caret)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(plotly)
#library(doParallel)
#registerDoParallel()


#-------------------------------------DATA PREPARATION----------------------------------------------

#setting working directory
#setwd("D:\\PGDDA\\Course 4 - PA2\\M3 - Handwritten letter recoginition")

# Loading Train data from csv file
train <- read.csv("mnist_train.csv")

#Giving names to column
colnames(train)[1]<-"Pattern_Rec"
for(i in seq(2,ncol(train),by=1)){colnames(train)[i]<-paste("Column",as.character(i-1),sep = " ")}

# Loading Test data from csv file
test <- read.csv("mnist_test.csv")

#Giving names to column
colnames(test)[1]<-"Pattern_Rec"
for(i in seq(2,ncol(test),by=1)){colnames(test)[i]<-paste("Column",as.character(i-1),sep = " ")}

#Duplicate exists or not
nrow(train[!duplicated(train), ])
# Since number of unique rows and rows in train data frame is 59999, So, no duplicate exists.
nrow(test[!duplicated(test), ])
# Since number of unique rows and rows in test data frame is 9999, So, no duplicate exists.

# Viewing the train and test Data set
view(train)
View(test)

#Checking the structure, dimentions and first few records of train and test dataset
str(train)
dim(train)   #59999 rows and 785 Columns
head(train)



# checking for blank values
sapply(train,function(x) length(which(x==" "))) 
# 0 blank values

# checking for missing values "NA" in dataset
sum(is.na(train))
sapply(train,function(x) length(which(is.na(x))))
# 0 NA values

str(test)
dim(test)       # 9999 rows and 785 columns
head(test)

# checking for blank values
sapply(test,function(x) length(which(x==" "))) 
# 0 blank values

# checking for missing values "NA" in dataset
sum(is.na(test))
sapply(test,function(x) length(which(is.na(x))))
# 0 NA values

# Converting Pattern_Rec column into factor for both dataset
train$Pattern_Rec <- factor(train$Pattern_Rec)
test$Pattern_Rec <- factor(test$Pattern_Rec)



#---------------------------------------MODEL BUILDING----------------------------------------------
set.seed(80)
train_index <- sample(1:nrow(train), 0.1*nrow(train))
train1 <- train[train_index, ]

test_index <- sample(1:nrow(test), 1*nrow(test))
test1 <- test[test_index, ]

# Scaling the columns
# All kernel methods are based on distance. Hence, it is required to scale our variables. 
# If we do not standardize our variables to comparable ranges, the variable with the largest range will completely dominate in the computation of the kernel matrix.
# Another reason of standardization is to avoid numerical difficulties during computation. 
# Because kernel values usually depend on the inner products of feature vectors, e.g. the linear kernel and the polynomial kernel, large attribute values might cause numerical problems.
# RGB values are encoded as 8-bit integers, which range from 0 to 255. 
# It's an industry standard to think of 0.0f as black and 1.0f as white (max brightness). 
# To convert [0, 255] to uniform standard value, it is good to divide by 255.

train1[, -1] <- train1[,-1]/255
test1[, -1] <- test1[, -1]/255
test[,-1] <- test[, -1]/255

#---------------------------------------  1.  LINEAR KERNEL ------------------------------------------
Model_linear <- ksvm(Pattern_Rec~ ., data = train1, scale = FALSE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, test1)
confusionMatrix(Eval_linear,test1$Pattern_Rec)
# Accuracy 0.9186
# Kappa 0.9095

#-----------HYPER-PARAMETER AND CROSS VALIDATION IN LINEAR MODEL-------------------------------
trainControl <- trainControl(method="cv", number=5)
metric <- "Accuracy"
set.seed(80)
Model_linear

# Making a grid of C value 
grid <- expand.grid(C=seq(1, 5, by=1))

# Performing 5 fold cross validation
fit_svm <- train(Pattern_Rec~., data=train1, method="svmLinear", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

# Printing cross validation result
print(fit_svm)
# Best tune at C=1, 
# Accuracy = 0.9048183
# Kappa = 0.8941750
plot(fit_svm)
# Plot clearly shows that C=1 is the best.
# We are getting good accuracy at cost C = 1.


#===========================Valdiating the model after cross validation on test data=======================

evaluate_linear<- predict(fit_svm, test)
plot_ly(x = ~evaluate_linear, type = "histogram")
plot_ly(x = ~evaluate_linear, type = "box")
confusionMatrix(evaluate_linear, test$Pattern_Rec)

# Accuracy after cross validation on test data    = 0.9186
# Kappa : 0.9095 
# Best tunning at C=1



#-----------------------------------  2.  RBF KERNEL  --------------------------------------------------
Model_RBF <- ksvm(Pattern_Rec~ ., data = train1,scale = FALSE, kernel = "rbfdot")
Evaluate_RBF<- predict(Model_RBF, test1)
confusionMatrix(Evaluate_RBF,test1$Pattern_Rec)

# Accuracy : 0.9512
# Kappa : 0.9457 

trainControl_rbf <- trainControl(method="cv", number=5)
metric <- "Accuracy"
set.seed(80)
Model_RBF

# cost C = 1
# Hyperparameter : sigma =  0.0107258687340991 
# Training error :  0.022837 

# Constructing grid of "sigma" and C values 
grid_rbf <- expand.grid(.sigma =c(0.025, 0.05), .C=c(0.1,1,2) )

# Performing 5-fold cross validation
fit.svm_rbf <- train(Pattern_Rec~., data=train1, method="svmRadial", metric=metric, 
                 tuneGrid=grid_rbf, trControl=trainControl_rbf)

# It is taking more than 20 min to execute because dataset contains 10K entries i.e. 10%
# If i take less entries than it will take less time but it is not good to train the model on less data.
# Here I am assuming that time is not the constraint but our model should be robust.



# Printing cross validation result
print(fit.svm_rbf)
# Best tune at sigma = 0.025 & C=2, Accuracy = 0.9611633 and Kappa = 0.9568251

# Plotting model results
plot(fit.svm_rbf)
# Above plot is clearly showing that lower Sigma value is perfoming good on CV.

#========================================  Checking overfitting - Non-Linear - SVM  ========================================================

# Validating on test data

evaluate_rbf<- predict(fit.svm_rbf, test)
plot_ly(x = ~evaluate_rbf, type = "histogram")
# This histogram shows that all the data is distributed uniformly.
plot_ly(x = ~evaluate_rbf, type = "box")

confusionMatrix(evaluate_rbf, test$Pattern_Rec)
# Accuracy : 0.9653 
# Kappa : 0.9614

# RBF is performing fairly good with Accuracy =  0.9653 and Kappa = 0.9614
# Optimal value for tunning is at sigma = 0.025 & C=2
# This model is performing good on unseen data too.


#============================================ 3. POLYNOMIAL KERNAL ====================================================

Model_poly <- ksvm(Pattern_Rec~ ., data = train1,scale = FALSE, kernel = "polydot")
Eval_poly<- predict(Model_poly, test1)
confusionMatrix(Eval_poly,test1$Pattern_Rec)

# Accuracy : 0.9186 and Kappa : 0.9095

trainControl_poly <- trainControl(method="cv", number=5)
metric <- "Accuracy"
set.seed(80)
Model_poly


# Cost C = 1
# Hyperparameters : degree =  1  scale =  1  offset =  1 
# Training error :   0.001334  


# Constructing grid of "sigma" and C values. 
grid_poly <- expand.grid(.degree = c(2,3), .scale = c(1,2), .C = c(0.1,0.5,1) )

# Performing 5-fold cross validation
fit.svm_poly <- train(Pattern_Rec~., data=train1, method="svmPoly", metric=metric, 
                     tuneGrid=grid_poly, trControl=trainControl_poly)



# Printing cross validation result
print(fit.svm_poly)
# The final values used for the model were degree = 2, scale = 1 and C = 0.1
# Accuracy = 0.9483261 and Kappa = 0.9425520

# Plotting model results
plot(fit.svm_poly)

#=================== Checking overfitting - Non-Linear - SVM ===================================

# Validating the model results on test data

evaluate_poly<- predict(fit.svm_poly, test)
plot_ly(x = ~evaluate_poly, type = "histogram")
plot_ly(x = ~evaluate_poly, type = "box")
confusionMatrix(evaluate_poly, test$Pattern_Rec)
# Accuracy : 0.955 
# Kappa : 0.95 

# With degree = 2, scale = 1 and C = 0.1, model is giving Accuracy of 0.955 and Kappa is 0.95


#***********************************  INSIGHTS AND OBSERVATION  ****************************************************
# 
#                       ACCURACY    KAPPA     COST(C)    SIGMA 
# LINEAR MODEL      :   0.9186      0.9095     1          -
# RBF MODEL         :   0.9653      0.9614     2         0.025
# POLYNOMIAL MODEL  :   0.955       0.95       0.1
# 
# 
# After Comparing above parameters, RBF Model is performing better than other two models n terms of
# accuracy, kappa and also having low non-linearity parameter i.e. Sigma.
#  - RBF model is well in accurately prredicting the digit of digital image in all different format.  
#  - Non linearity parameter i.e. Sigma is very low i.e. 0.025
#  - Even though, it is little complex, but performing fairly good.
# 
# * Although, other 2 models are also significant and performing good with mnist dataset. 



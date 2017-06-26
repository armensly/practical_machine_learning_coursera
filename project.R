library(caret)
setwd("~/coursera/coursera_practical_machine_learning/project/")
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
              destfile = "pml-training.csv", method = "curl")
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
              destfile = "pml-testing.csv", method = "curl")
### splitting the data:
set.seed(123)
data <- read.csv("pml-training.csv", na.strings=c("NA", "#DIV/0!", " "))
inTrain <- createDataPartition(data$classe, p = 0.6, list = FALSE)
training <- data[inTrain,]
testing <- data[-inTrain,]
inTest <- createDataPartition(testing$classe, p = 0.5, list = FALSE)
validation <- testing[-inTest,]
testing <- testing[inTest,]

### removing unwanted columns from the data:
training <- training[,-(1:7)]
fullCols <- (colSums(is.na(training)) == 0)
training <- training[, fullCols]

### training on different models:
#using 25 bootstraping samples
modelBoot <- train(classe~., data = training, method = "rf", ntree = 100, trControl = trainControl(method="boot"))
#using 25 bootstrap632
modelBoot632 <- train(classe~., data = training, method = "rf", ntree = 100, trControl = trainControl(method="boot632"))
#using 10-fold cross-validation on mtry (randomly selected predictors)
modelCv <- train(classe~., data = training, method = "rf", ntree = 100, trControl = trainControl(method="cv"))
#using 10-fold repeated cross-validation (10 repetition)
modelRepCv <- train(classe~., data = training, method = "rf", ntree = 100, trControl = trainControl(method="repeatedcv"))
#using larger number of trees
modelBoot200 <- train(classe~., data = training, method = "rf", ntree = 200, trControl = trainControl(method="boot"))
#using classification tree (rpart) with cross-validation for complexity parameter (depth of tree)
modelRpart <- train(classe~., data = training, method = "rpart", trControl = trainControl(method = "cv"))
#using gbm
modelGbm <- train(classe~., data = training, method = "gbm")
#using linear discreminant analysis
modelLda <- train(classe~., data = training, method = "lda")
#with pca
prep <- preProcess(training, method = c("pca"))
trainingPca <- predict(prep, training)
validationPca <- predict(prep, validation)
testingPca <- predict(prep, testing)
modelBootPca <- train(classe~., data = trainingPca, method = "rf", ntree = 100)
save.image("half_learned.Rdata")

### validation on different models:
res <- data.frame(method=c(), acc=c())
res <- rbind(res, data.frame(method="rf_boosting", 
                             acc=confusionMatrix(predict(modelBoot, validation), validation$classe)$overall[1]))
res <- rbind(res, data.frame(method="rf_boosting632", 
                             acc=confusionMatrix(predict(modelBoot632, validation), validation$classe)$overall[1]))
res <- rbind(res, data.frame(method="rf_cv", 
                             acc=confusionMatrix(predict(modelCv, validation), validation$classe)$overall[1]))
res <- rbind(res, data.frame(method="rf_repeatedCV", 
                             acc=confusionMatrix(predict(modelRepCv, validation), validation$classe)$overall[1]))
res <- rbind(res, data.frame(method="rf_boosting200tree", 
                             acc=confusionMatrix(predict(modelBoot200, validation), validation$classe)$overall[1]))
res <- rbind(res, data.frame(method="rpart", 
                             acc=confusionMatrix(predict(modelRpart, validation), validation$classe)$overall[1]))
res <- rbind(res, data.frame(method="GBM", 
                             acc=confusionMatrix(predict(modelGbm, validation), validation$classe)$overall[1]))
res <- rbind(res, data.frame(method="LDA", 
                             acc=confusionMatrix(predict(modelLda, validation), validation$classe)$overall[1]))
res <- rbind(res, data.frame(method="rf_boosting_w_PCA", 
                             acc=confusionMatrix(predict(modelBootPca, validationPca), validation$classe)$overall[1]))
res

###
save.image(file = "half_learned.Rdata")
v <- varImp(modelBoot)
v
selectedFeatures <- order(v$importance$Overall, decreasing = TRUE)[1:7]
selectTraining <- training[, selectedFeatures]
selectTraining <- cbind(selectTraining, classe=training$classe)
modelBootSelect <- train(classe~., data = selectTraining, method = "rf", ntree = 100, trControl = trainControl(method="boot"))
res <- rbind(res, data.frame(method="rf_boost_selectFeatures", 
                             acc=confusionMatrix(predict(modelBootSelect, validation), validation$classe)$overall[1]))
featurePlot(x = training[,selectedFeatures], y = training$classe)

library(ggplot2)
ggplot(data = training, aes(roll_belt, pitch_forearm, fill = classe, color = classe)) + geom_rug()

### final testing for the quiz:
final_testing <- read.csv("pml-testing.csv", na.strings=c("NA", "#DIV/0!", " "))

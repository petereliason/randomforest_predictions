
# Loading package
library(caTools)
library(randomForest)

mydata <- read.csv("pima-indians-diabetes.csv")
mydata$Y <- as.factor(mydata$Y)

# Splitting data in train and test data
split <- sample.split(mydata, SplitRatio = 0.7)
split

train <- subset(mydata, split == "TRUE")
test <- subset(mydata, split == "FALSE")

# Fitting Random Forest to the train dataset
set.seed(120) # Setting seed
X = train[-9]
classifier_RF = randomForest(x = X,
                             y = train$Y,
                             ntree = 40)

classifier_RF

# Predicting the Test set results
y_pred = predict(classifier_RF, newdata = test[-9])

# Confusion Matrix
confusion_mtx = table(test[, 9], y_pred)
confusion_mtx
accuracy = (sum(diag(confusion_mtx)))/sum(confusion_mtx)
print(accuracy)
# Plotting model
plot(classifier_RF)

# Importance plot
importance(classifier_RF)

# Variable importance plot
varImpPlot(classifier_RF)

# makes the random forest submission

library(randomForest)

train <- read.csv("../data/train.csv", header=TRUE)
test <- read.csv("../data/test.csv", header=TRUE)

labels <- as.factor(train[,1])
train <- train[,-1]

	print(Sys.time())
	
rf <- randomForest(train, labels, xtest=test, ntree=2,mtry=2, importance=FALSE, na.action=na.omit,keep.forest = FALSE)

print(rf$test$predicted)
quit()
predictions <- levels(labels)[rf$test$predicted]

	print(Sys.time())
	
write(predictions, file="../predictions/rf_benchmark.csv", ncolumns=1) 

require(nnet)
require(caret)
library("randomForest")
#library(mlbench)


Xtrain <- as.matrix(read.table("PreProcessData\\training_PreProcess4_40.csv",
                   header=FALSE,
                   sep= "\t"       
                   ))
Weights <- as.matrix(read.table("PreProcessData\\weights.csv",
                   header=FALSE,
                   sep= " "       
                   ))		   
Xtest <- as.matrix(read.table("PreProcessData\\test_PreProcess4_40.csv",
                   header=FALSE,
                   sep= "\t"       
                   ))
				   
				   
print("Loaded Data")


dim(Xtrain)
dim(Xtest)

X <- Xtrain

dim (X)

#sample.int(10)
#quit()

folds <- createFolds(X[,1], k = 5)

CVFolds = 5
trainPredictions <- rep(NA, nrow(X)) 
testPredictions <- matrix(data=NA,nrow=nrow(Xtest),ncol=CVFolds)

for(i in 1:CVFolds) {
        
	print(paste("Fold dim: ", as.character(i)))
	
	# these are the real one for CV.
	YY <- X[-folds[[i]],1]	
	XX <- X[-folds[[i]],-1]

	#YY <- X[1:30000,1]	
	#XX <- X[1:30000,-1]
	
	#print(dim(YY))
	#print(dim(XX))

	#tuneRF(x, y, mtryStart, ntreeTry=50, stepFactor=2, improve=0.05,trace=TRUE, plot=TRUE, doBest=FALSE, ...)
	# ntree=75 is 3.12
	# ntree 100 is mtry = 10 is 3.13
	# ntree 75, mtry 20 is 3.12
	# try mtry lower than 10 and ntree higher than 75.
	# try this one...
	
	#XXtest <- X[folds[[i]],-1]
	XXtest <- Xtest
	print(Sys.time())
	lm.fit <- randomForest(x=XX, y=YY,xtest=XXtest, ntree=10, mtry=5, importance=FALSE, na.action=na.omit,keep.forest = FALSE) #keep.forest = FALSE helps memory
	print(Sys.time())
	#print(lm.fit$test$predicted)
	
	currScore = sum(Weights[folds[[i]]] * abs(lm.fit$test$predicted - X[folds[[i]],1])) / sum(Weights[folds[[i]]]) # pred*100.0
	
	print(lm.fit$test$predicted)
	print("score:")
	print (currScore)
	
	
	quit()
	
	
	#lm.fit <- nnet(YY/100.0 ~ XX, size = 10, rang = 0.1, decay = 5e-2, maxit = 300) # size = 7, rang = 0, decay = 0.1, maxit = 500
	#lm.fit <- lm(YY ~ XX) 

	#YY <- Y[folds[[i]]]
	XX <- X[folds[[i]],-1]	
	print(colnames(XX))
	cNames <- colnames(XX)
	pred <-predict(lm.fit,data.frame(XX)) #, se.fit= TRUE
	
	#predictions[folds[[i]]]<- pred$fit  # for LM
	trainPredictions[folds[[i]]]<- pred  #*100.0
	

	currScore = sum(Weights[folds[[i]]] * abs(pred - X[folds[[i]],1])) / sum(Weights[folds[[i]]]) # pred*100.0
	print (currScore)
	
	#system.time(x <- matrix(data=NA,nrow=10000,ncol=10000))	
		
	XX <- Xtest	
	colnames(XX) <- cNames

	pred <-predict(lm.fit,data.frame(XX)) 		#, se.fit= TRUE
	testPredictions[,i]<- pred  #*100.0
				
	#memory.size(), memory.limit()memory.profile()
 }

 
 
write.table(rowMeans(testPredictions), file="temp\\pred_R_Test.txt", row.names = FALSE, col.names = FALSE)   

write.table(trainPredictions, file="temp\\pred_R_Train.txt", row.names = FALSE, col.names = FALSE) #, row.names = FALSE, col.names = FALSE
print(sum(Weights * abs(trainPredictions - X[,1])) / sum(Weights))

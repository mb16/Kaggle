require(nnet)
require(caret)
require(gam)
#library(mlbench)


X <- as.matrix(read.table("temp\\dataset_blend_trainX.txt",
                   header=FALSE,
                   sep= " "       
                   ))
Y <- as.matrix(read.table("temp\\dataset_blend_trainY.txt",
                   header=FALSE,
                   sep= " "       
                   ))
Weights <- as.matrix(read.table("PreProcessData\\weights.csv",
                   header=FALSE,
                   sep= " "       
                   ))		   
Xtest <- as.matrix(read.table("temp\\dataset_blend_testX.txt",
                   header=FALSE,
                   sep= " "       
                   ))
				   
				   
print("Loaded Data")

dim(X)
dim(Y)


folds <- createFolds(Y, k = 5)

CVFolds = 5
trainPredictions <- rep(NA, nrow(X)) 
testPredictions <- matrix(data=NA,nrow=nrow(Xtest),ncol=CVFolds)

for(i in 1:CVFolds) {
        
	print(paste("Fold dim: ", as.character(i)))

	
	YY <- Y[-folds[[i]]]
	XX <- X[-folds[[i]],]	
	
	# note, gam is slightly better than the neural network.
	lm.fit <- gam(YY/100.0 ~ XX, family = gaussian, trace=TRUE) #family = binomial, for classification??
	#lm.fit <- nnet(YY/100.0 ~ XX, size = 2, rang = 0.1, decay = 5e-3, maxit = 300) # size = 7, rang = 0, decay = 0.1, maxit = 500
	#lm.fit <- lm(YY ~ XX) 

	YY <- Y[folds[[i]]]
	XX <- X[folds[[i]],]	
	pred<-predict(lm.fit, newdata=data.frame(XX), type="response")
	#pred <-predict(lm.fit,data.frame(XX), se.fit= TRUE) 
	
	#predictions[folds[[i]]]<- pred$fit  # for LM
	trainPredictions[folds[[i]]]<- pred*100.0
	

	currScore = sum(Weights[folds[[i]]] * abs(pred*100.0 - Y[folds[[i]]])) / sum(Weights[folds[[i]]])
	print (currScore)
	
	#system.time(x <- matrix(data=NA,nrow=10000,ncol=10000))	
		
	XX = Xtest	
	pred<-predict(lm.fit, newdata=data.frame(XX), type="response")
	#pred<- pmax(0.01,pred)
	#pred<- pmin(0.98,pred)
	
	#pred <-predict(lm.fit,data.frame(XX), se.fit= TRUE) 		
	testPredictions[,i]<- pred*100.0

		
 }

 
 
write.table(rowMeans(testPredictions), file="temp\\pred_LMTest.txt", row.names = FALSE, col.names = FALSE)   

write.table(trainPredictions, file="temp\\pred_LMFinal.txt", row.names = FALSE, col.names = FALSE) #, row.names = FALSE, col.names = FALSE
print(sum(Weights * abs(trainPredictions - Y)) / sum(Weights))

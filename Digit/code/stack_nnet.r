require(nnet)
require(caret)
library("randomForest")
#library(mlbench)


Xtrain <- as.data.frame(read.table("../data/train.csv",
                   header=TRUE,
                   sep= ","       
                   ))
Xtest <- as.data.frame(read.table("../data/test.csv",
                   header=TRUE,
                   sep= ","       
                   ))
				   
				   
print("Loaded Data")


#names(Xtrain)
#summary(Xtrain)

dim(Xtrain)
dim(Xtest)

X <- Xtrain






trainPredictions <- matrix(data=0,nrow=nrow(X),ncol=10)
trainPredictionsSubtract <- rep(1, nrow(X)) 
testPredictions <- matrix(data=0,nrow=nrow(Xtest),ncol=10) # ncol=CVFolds



#sink("record.lis")

#gctorture(on = TRUE)
#gctorture2(1, wait = 1, inhibit_release = FALSE)

CVFolds = 2
Repeats <- 10

for(j in 1:Repeats) {
	print(paste("Repeat: ", as.character(j)))
	folds <- createFolds(X[,1], k = CVFolds)
	setScore = 0.0
	for(i in 1:CVFolds) {
			
		gc()	
			
		print(paste("Fold dim: ", as.character(i)))
		
		# these are the real one for CV.
		YY <- X[-folds[[i]],1]	
		XX <- X[-folds[[i]],-1]

		
		#print(cbind( Freq=table(YY), Cumul=cumsum(table(YY)), Relative=prop.table(table(YY))))

		#XXtest <- X[folds[[i]],-1]	
		print(Sys.time())
		#RandF <- randomForest(x=XX, y=as.factor(YY), ntree=100, importance=FALSE, na.action=na.omit,keep.forest = TRUE) #keep.forest = FALSE helps memory
		#nn <- nnet(x=XX, y=class.ind(YY), size = 2, MaxNWts = 20000, rang = 0.1,decay = 5e-4, maxit = 100)
		#print(colnames(Xtrain))
		
		Xtemp <- Xtrain[-folds[[i]],]
		Xtemp$label = class.ind(Xtemp$label)
		nn <- nnet(label ~ ., data=Xtemp, size = 20, MaxNWts = 40000, rang = 0.1,decay = 5e-4, maxit = 100)
		#nn <- nnet(XX, class.ind(YY), size = 50, MaxNWts = 40000, rang = 0.1,decay = 5e-4, maxit = 100)
		print(Sys.time())


		gc()	

		Xtemp <- as.data.frame(Xtrain[folds[[i]],-1])
		#print(colnames(XX))
		#cNames <- colnames(XX)
		XXtest <- X[folds[[i]],-1]	
		pred <-predict(nn ,Xtemp) #, type="class"

		print(pred)
		print("COLS")
		print(nrow(pred))
		quit()
		
		#predictions[folds[[i]]]<- pred$fit  # for LM
		#trainPredictions[folds[[i]]]<- as.integer(pred) # for RF class prediction
		k <- apply(pred, 1, function(x) max(which(x == max(x, na.rm = TRUE))) - 1 )
		trainPredictions[folds[[i]],] <- trainPredictions[folds[[i]],] + as.matrix(k)

		currScore = as.double(sum(ifelse(k == X[folds[[i]],1], 1, 0)))/as.double(nrow(XX))
		print (currScore)	
		setScore = setScore + currScore/as.double(CVFolds)
		
		quit()
		
		gc()	
		XX <- Xtest	
		colnames(XX) <- cNames
		pred <-predict(RandF,data.frame(XX), type="vote", norm.votes=FALSE) 		#, se.fit= TRUE
		testPredictions = testPredictions + as.matrix(pred)
		#testPredictions[,i]<- pred
					
		#memory.size(), memory.limit()memory.profile()
	 }

	gc()	
	print("Avg. Score")
	print(setScore)

	# note, must subtract 1 because indeices range 1 - 10, and our predictions range 0 - 9.
	k <- apply(testPredictions, 1, function(x) max(which(x == max(x, na.rm = TRUE))) - 1 ) # get max index of max value by row (1 in apply func)
	write.table(k, file="temp\\pred_R_Test.txt", row.names = FALSE, col.names = FALSE)   

	# must subtract 1 from all values because the indeices range 1 - 10, and out predictions range 0 - 9.
	m <- trainPredictions # - trainPredictionsSubtract
	k <- apply(trainPredictions, 1, function(x) max(which(x == max(x, na.rm = TRUE))) - 1 )
	write.table(k, file="temp\\pred_R_Train.txt", row.names = FALSE, col.names = FALSE) #, row.names = FALSE, col.names = FALSE


	finalScore = as.double(sum(ifelse(k  == Xtrain[,1], 1, 0)))/as.double(nrow(Xtrain))
	print("Cumulative Score")
	print (finalScore)


}
require(caret)
library("e1071")


Xtrain <- as.matrix(read.table("../data/train.csv",
                   header=TRUE,
                   sep= ","       
                   ))
Xtest <- as.matrix(read.table("../data/test.csv",
                   header=TRUE,
                   sep= ","       
                   ))
				   
				   
print("Loaded Data")


dim(Xtrain)
dim(Xtest)

X <- Xtrain



CVFolds = 10
Repeats <- 10
Classes = 10

#trainPredictions <- matrix(data=0,nrow=nrow(X),ncol=Classes)
trainPredictionsSubtract <- rep(1, nrow(X)) 
#testPredictions <- matrix(data=0,nrow=nrow(Xtest),ncol=Classes) # ncol=CVFolds

print("Loading Previous Training/Test Matrix...")

trainPredictions <- as.matrix(read.table("temp/trainPredictions_matrix_rf.txt",
                   header=FALSE,
                   sep= " "       
                   ))
testPredictions <- as.matrix(read.table("temp/testPredictions_matrix_rf.txt",
                   header=FALSE,
                   sep= " "       
                   ))


#sink("record.lis")

#gctorture(on = TRUE)
#gctorture2(1, wait = 1, inhibit_release = FALSE)

# for tuning need to always use the same fold.
folds <- createFolds(X[,1], k = CVFolds)
for(j in 1:Repeats) {
	print(paste("Repeat: ", as.character(j)))
	#folds <- createFolds(X[,1], k = CVFolds)
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
		SVMmodel <- svm(x=XX, y=as.factor(YY),kernel="polynomial", degree=3, coef0=0, cost=1, nu=0.5, class.weights=NULL,cachesize=40, tolerance=0.001, epsilon=0.1, shrinking=TRUE, cross=0, fitted=TRUE, seed=5) # , gamma=1/dim(x)[2]
		print(Sys.time())
		print(paste("MTRY: ", as.character(j*50)))

		gc()	
		#lm.fit <- nnet(YY/100.0 ~ XX, size = 10, rang = 0.1, decay = 5e-2, maxit = 300) # size = 7, rang = 0, decay = 0.1, maxit = 500
		#lm.fit <- lm(YY ~ XX) 

		#YY <- Y[folds[[i]]]
		XX <- X[folds[[i]],-1]	
		#print(colnames(XX))
		cNames <- colnames(XX)
		pred <-predict(SVMmodel,data.frame(XX))

		print(pred)
		print("COLS")
		print(nrow(pred))
		quit()
		
		#predictions[folds[[i]]]<- pred$fit  # for LM
		#trainPredictions[folds[[i]]]<- as.integer(pred) # for RF class prediction
		trainPredictions[folds[[i]],] <- trainPredictions[folds[[i]],] + as.matrix(pred)

		k <- apply(pred, 1, function(x) max(which(x == max(x, na.rm = TRUE))) - 1 )
		currScore = as.double(sum(ifelse(k == X[folds[[i]],1], 1, 0)))/as.double(nrow(XX))
		print (currScore)	
		setScore = setScore + currScore/as.double(CVFolds)
		
		gc()	
		XX <- Xtest	
		colnames(XX) <- cNames
		pred <-predict(SVMmodel,data.frame(XX), type="vote", norm.votes=FALSE) 		#, se.fit= TRUE
		testPredictions = testPredictions + as.matrix(pred)
		#testPredictions[,i]<- pred
					
		#memory.size(), memory.limit()memory.profile()
		break
	 }

	gc()	
	print("Avg. Score")
	print(setScore)

	# note, must subtract 1 because indeices range 1 - 10, and our predictions range 0 - 9.
	k <- apply(testPredictions, 1, function(x) max(which(x == max(x, na.rm = TRUE))) - 1 ) # get max index of max value by row (1 in apply func)
	write.table(k, file="temp\\pred_R_Test.txt", row.names = FALSE, col.names = FALSE)   

	write.table(testPredictions, file="temp\\testPredictions_matrix_svm.txt", row.names = FALSE, col.names = FALSE)  
	
	# must subtract 1 from all values because the indeices range 1 - 10, and out predictions range 0 - 9.
	m <- trainPredictions # - trainPredictionsSubtract
	k <- apply(trainPredictions, 1, function(x) max(which(x == max(x, na.rm = TRUE))) - 1 )
	write.table(k, file="temp\\pred_R_Train.txt", row.names = FALSE, col.names = FALSE) #, row.names = FALSE, col.names = FALSE

	write.table(trainPredictions, file="temp\\trainPredictions_matrix_svm.txt", row.names = FALSE, col.names = FALSE) #, row.names = FALSE, col.names = FALSE

	finalScore = as.double(sum(ifelse(k  == Xtrain[,1], 1, 0)))/as.double(nrow(Xtrain))
	print("Cumulative Score")
	print (finalScore)


}
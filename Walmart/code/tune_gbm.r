require(gbm)
require(caret)

Xtrain <- as.matrix(read.table("train.csv",
                   header=TRUE,
                   sep= ","       
                   ))
Xtest <- as.matrix(read.table("test.csv",
                   header=TRUE,
                   sep= ","       
                   ))
				   
				   
print("Loaded Data")


#names(Xtrain)
#summary(Xtrain)
#dim(Xtrain)
#dim(Xtest)




trainPredictions <- matrix(data=0,nrow=nrow(Xtrain),ncol=1)


bestScore = 500000.0

CVFolds = 2
Repeats <- 1
targetCOlumn = 5
weightColumn = 17
features = "all"

folds <- createFolds(Xtrain[,targetCOlumn], k = CVFolds)

for(repeats in 1:Repeats) {
	#print(paste("Repeat: ", as.character(j)))
  
  #folds <- createFolds(X[,1], k = CVFolds)
  for(cvFolds in 1:CVFolds) {



		YY <- Xtrain[-folds[[cvFolds]],targetCOlumn]	
		XX <- Xtrain[-folds[[cvFolds]],-targetCOlumn]
    
    
		print(Sys.time())

		#SVMmodel <- svm(x=XX, y=as.numeric(YY),type="eps-regression", kernel="polynomial", degree=3, coef0=0, cost=1, nu=0.5, class.weights=NULL,cachesize=40, tolerance=0.001, epsilon=0.1, shrinking=TRUE, cross=0, fitted=TRUE, seed=5) # , gamma=1/dim(x)[2]
		
		gbm1 <-
		  gbm.fit(x=XX, y=as.numeric(YY),
		      #var.monotone=c(0,0,0,0,0,0), # -1: monotone decrease, # +1: monotone increase,# 0: no monotone restrictions
		      distribution="gaussian", # see the help for other choices
		      n.trees=200, # number of trees
		      shrinkage=0.05, # shrinkage or learning rate,   # 0.001 to 0.1 usually work
		      interaction.depth=3, # 1: additive model, 2: two-way interactions, etc.
		      bag.fraction = 0.5, # subsampling fraction, 0.5 is probably best
		      #train.fraction = 1.0, # fraction of data for training,12 gbm
		      # first train.fraction*N used for training
		      n.minobsinnode = 10, # minimum total weight needed in each node
		      #cv.folds = 1, # do 3-fold cross-validation
		      keep.data=TRUE, # keep a copy of the dataset with the object
		      verbose=FALSE, # don't print out progress
		      #n.cores=1
          ) # use only a single core (detecting
    
    
		#best.iter <- gbm.perf(gbm1,method="OOB")
		#print(best.iter)
		
    
    
		pred <-predict(gbm1,XX,n.trees=200)
  
		currScore = sum(abs(pred - YY) * Xtrain[,weightColumn]) / sum(Xtrain[,weightColumn])
		print (currScore)	
    

		if (currScore < bestScore ) {
			print("Found Best Score: Saving New Files")
			bestScore <- currScore
			filename <- paste("..\\predictions\\Target_Stack_", format(Sys.time(), '%Y%m%d%H%M%S'), "_", as.character(currScore), "_", "knn", ".csv", sep = "")
			write.table(YY, file=filename, row.names = FALSE, col.names = FALSE) #, row.names = FALSE, col.names = FALSE

			#results <- knn(train=XX, test=XtestPCA[,1:features], cl= YY, k = neighbors, algorithm="VR")
			
			filename <- paste("..\\predictions\\Stack_", format(Sys.time(), '%Y%m%d%H%M%S'), "_", as.character(currScore), "_", "knn", ".csv", sep = "")
			write.table(pred, file=filename, row.names = FALSE, col.names = FALSE)   

			
			cat("GBM" ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 
			cat("Features:", as.character(features) ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 			
			#cat("Neighbors:", as.character(neighbors) ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 	
			cat("Score:", as.character(bestScore) ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 	
			cat("DateTime:", format(Sys.time(), '%Y %m %d %H:%M:%S') ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 	
			cat("DateTime:", format(Sys.time(), '%Y %m %d %H:%M:%S') ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 				
			cat("\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 
		
    }
	 }
	 

	 
}
	 
	 





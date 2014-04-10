require(caret)
library("e1071")

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

CVFolds = 10
Repeats <- 10
targetCOlumn = 5
weightColumn = 17
features = "all"

folds <- createFolds(Xtrain[,targetCOlumn], k = CVFolds)

for(repeats in 1:Repeats) {
	#print(paste("Repeat: ", as.character(j)))
  
  #folds <- createFolds(X[,1], k = CVFolds)
  for(cvFolds in 1:CVFolds) {


		print(paste("Neighbors:", as.character(neighbors)))
		# these are the real one for CV.


		YY <- Xtrain[-folds[[i]],targetCOlumn]	
		XX <- Xtrain[-folds[[i]],-targetCOlumn]
    
    
		print(Sys.time())

		SVMmodel <- svm(x=XX, y=as.numeric(YY),type="eps-regression", kernel="polynomial", degree=3, coef0=0, cost=1, nu=0.5, class.weights=NULL,cachesize=40, tolerance=0.001, epsilon=0.1, shrinking=TRUE, cross=0, fitted=TRUE, seed=5) # , gamma=1/dim(x)[2]
		
		pred <-predict(SVMmodel,XX)
    print(names(pred))
  
    
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

			
			cat("KNN" ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 
			cat("Features:", as.character(features) ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 			
			cat("Neighbors:", as.character(neighbors) ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 	
			cat("Score:", as.character(bestScore) ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 	
			cat("DateTime:", format(Sys.time(), '%Y %m %d %H:%M:%S') ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 	
			cat("DateTime:", format(Sys.time(), '%Y %m %d %H:%M:%S') ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 				
			cat("\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 
		
    }
	 }
	 

	 
}
	 
	 





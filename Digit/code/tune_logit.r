
require(caret)
library("caTools")



Xtrain <- as.matrix(read.table("../data/train.csv",
                   header=TRUE,
                   sep= ","       
                   ))
Xtest <- as.matrix(read.table("../data/test.csv",
                   header=TRUE,
                   sep= ","       
                   ))
				   
				   
print("Loaded Data")


#names(Xtrain)
#summary(Xtrain)

#dim(Xtrain)
#dim(Xtest)


trainPredictions <- matrix(data=0,nrow=nrow(Xtrain),ncol=1)

sink("record.lis")



#pc.cr <- prcomp(x=Xtrain[,-1]) # , scale. =TRUE, retx=TRUE
#print("PCA Done")
#XtrainPCA <- predict(pc.cr, newdata=Xtrain[,-1])
#print("PCA Predict train")
#XtestPCA <- predict(pc.cr, newdata=Xtest)

#print(summary(pc.cr))
#print(summary(train))
#print(summary(Xtest))
#print("Done PCA")
gc()	


#gctorture(on = TRUE)
#gctorture2(1, wait = 1, inhibit_release = FALSE)

bestScore = 0.0
CVFolds = 5

folds <- createFolds(Xtrain[,1], k = CVFolds)

for(j in 1:1) {
	#print(paste("Repeat: ", as.character(j)))

	for(i in 1:1) {
			
		gc()	
				

		YY <- Xtrain[-folds[[i]],1]	
		XX <- Xtrain[-folds[[i]],-1]

		#YY <- Xtrain[folds[[i]],1]	
		#XX <- Xtrain[folds[[i]],-1]
		
		#print(cbind( Freq=table(YY), Cumul=cumsum(table(YY)), Relative=prop.table(table(YY))))


		print(Sys.time())
		# 0.85 score for ncol(XX), 0.87 for ncol(XX)/2, 0.86 for ncol(XX)/4,  seems like there is nothing to tune here...
		LB <- LogitBoost(xlearn=XX, ylearn=YY, nIter=ncol(XX)/4+ 1)
		
		print(Sys.time())

		XX <- Xtrain[folds[[i]],-1]	
		YY <- Xtrain[folds[[i]],1]	
		#XX <- Xtrain[-folds[[i]],-1]	
		#YY <- Xtrain[-folds[[i]],1]	
		results <- predict(LB, XX, type="raw")
		print(results)
		#view data, and figure out how to find column with max votes, and split ties.
		k <- apply(results, 1, function(x) max(which(x == max(x, na.rm = TRUE))) - 1 )
		print(k)
		currScore = as.double(sum(ifelse(k == YY, 1, 0)))/as.double(nrow(XX))
		print (currScore)	
		
		quit()
		
		if ( currScore > bestScore | currScore > 0.9740) {
			print("Found Best Score: Saving New Files")
			bestScore <- currScore
			filename <- paste("..\\predictions\\Target_Stack_", format(Sys.time(), '%Y%m%d%H%M%S'), "_", as.character(currScore), "_", "logit", ".csv", sep = "")
			write.table(trainPredictions, file=filename, row.names = FALSE, col.names = FALSE) #, row.names = FALSE, col.names = FALSE

			print(Sys.time())
			LB <- LogitBoost(xlearn=XX, ylearn=YY, nIter=ncol(XX)+ 1)
			print(Sys.time())
			results <- predict(LB, XX, type="raw")
			k <- apply(results, 1, function(x) max(which(x == max(x, na.rm = TRUE))) - 1 )
						
			filename <- paste("..\\predictions\\Stack_", format(Sys.time(), '%Y%m%d%H%M%S'), "_", as.character(currScore), "_", "knn", ".csv", sep = "")
			write.table(k, file=filename, row.names = FALSE, col.names = FALSE)   

			
			cat("LogitBoost" ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 
			cat("Score:", as.character(bestScore) ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 	
			cat("DateTime:", format(Sys.time(), '%Y %m %d %H:%M:%S') ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 	
			cat("DateTime:", format(Sys.time(), '%Y %m %d %H:%M:%S') ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 				
			cat("\r\n", file = "..\\predictions\\RunLog.csv",sep="\r\n") 
		}
	 }
	 

	 
}
	 
	 






library(FNN)
require(caret)
library("randomForest")



Xtrain <- as.matrix(read.table("../data/train.csv",
                   header=TRUE,
                   sep= ","       
                   ))
Xtest <- as.matrix(read.table("../data/test.csv",
                   header=TRUE,
                   sep= ","       
                   ))
				   
				   
print("Loaded Data")


names(Xtrain)
#summary(Xtrain)

dim(Xtrain)
dim(Xtest)

#X <- Xtrain


trainPredictions <- matrix(data=0,nrow=nrow(Xtrain),ncol=1)

#sink("record.lis")


pc.cr <- prcomp(x=Xtrain[,-1]) # , scale. =TRUE, retx=TRUE
print("PCA Done")
XtrainPCA <- predict(pc.cr, newdata=Xtrain[,-1])
print("PCA Predict train")
XtestPCA <- predict(pc.cr, newdata=Xtest)

gc()	

#YY <- Xtrain[,1]	
#XX <- XtrainPCA[,1:(40)]
#results <- knn(train=XX, test=XtestPCA[,1:(40)], cl= YY, k = 3, algorithm="cover_tree")

#write.table(results, file="temp\\pred_R_Test_knn.txt", row.names = FALSE, col.names = FALSE)   
#quit()

#print(summary(pc.cr))
#print(summary(train))
#print(summary(Xtest))
print("Done PCA")
gc()	


#gctorture(on = TRUE)
#gctorture2(1, wait = 1, inhibit_release = FALSE)

CVFolds = 10
for(j in 1:1) {
	#print(paste("Repeat: ", as.character(j)))
	folds <- createFolds(Xtrain[,1], k = CVFolds)
	setScore = 0.0
	for(i in 1:CVFolds) {
			
		gc()	
			
		print(paste("Fold dim: ", as.character(i)))
		
		print(paste("Features: ", as.character(40)))
		# these are the real one for CV.
		YY <- Xtrain[-folds[[i]],1]	
		#XX <- Xtrain[-folds[[i]],-1]
		XX <- XtrainPCA[-folds[[i]],1:(40)] #80
		
		#print(cbind( Freq=table(YY), Cumul=cumsum(table(YY)), Relative=prop.table(table(YY))))

		#XXtest <- Xtrain[folds[[i]],-1]	
		XXtest <- XtrainPCA[folds[[i]],1:(40)]	# 80
		print(Sys.time())

		#neighbors = j
		neighbors = 3  # seems best...
		print(neighbors)
		# from neighbors 2 - 8, 8 is best (0.9671).  check 8 - 16
		# 8 seems to be the best...
		
		# all using "cover_tree"
		# with pca-80 and knn-10, 0.9696
		# with PCA 80 , 1 = 0.9723, 2=0.9672, 3=9715, 4=9710, 5=9695, 6=9728, 7=9702, 8=9695, 9=9675, 10=9685
		# with PCA  40 features and neighbors 6 => 0.9741
		# with PCA  80 features and neighbors 6 => 0.9727
		# with PCA 40 and neighbors 4, => 0.9745, 6 was 0.9726 this run. depends of CV selection.
		# with PCA 40 1 => 9726, 2 =>  9690, 3 => 9737  (cv 10 was 97445), 4 => 9732, 5 => 9735, 6 => 9735, 7 => 9720, 8 => 9724, (5 cv)
		results <- knn(train=XX, test=XXtest, cl= YY, k = neighbors, algorithm="VR")

		#print(results)
		print(Sys.time())

		currScore = as.double(sum(ifelse(results == Xtrain[folds[[i]],1], 1, 0)))/as.double(nrow(XXtest))
		print (currScore)	
		setScore = setScore + currScore/as.double(CVFolds)
		
		
		trainPredictions[folds[[i]],] <- results

		
	 }
	 
	gc()	
	print("Avg. Score")
	print(setScore)
	 
}
	 
	 
	write.table(trainPredictions, file="temp\\pred_R_Train_knn.txt", row.names = FALSE, col.names = FALSE) #, row.names = FALSE, col.names = FALSE

	 	
	#YY <- Xtrain[,1]	
	#XX <- Xtrain[,-1]
	YY <- XtrainPCA[,1]	
	XX <- XtrainPCA[,1:(40)]
	results <- knn(train=XX, test=XtestPCA[,1:(40)], cl= YY, k = 3, algorithm="cover_tree")

	write.table(results, file="temp\\pred_R_Test_knn.txt", row.names = FALSE, col.names = FALSE)   





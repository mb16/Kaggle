
library(FNN)
require(caret)
library("nnet")


# use half the iris data

ir <- rbind(iris3[,,1],iris3[,,2],iris3[,,3])

targets <- class.ind( c(rep("s", 50), rep("c", 50), rep("v", 50)) )

samp <- c(sample(1:50,25), sample(51:100,25), sample(101:150,25))
print(samp)
ir1 <- nnet(ir[samp,], targets[samp,], size = 2, rang = 0.1,
decay = 5e-4, maxit = 200)
test.cl <- function(true, pred) {
	true <- max.col(true)
	cres <- max.col(pred)
	table(true, cres)
}
test.cl(targets[-samp,], predict(ir1, ir[-samp,]))

dat <- predict(ir1, ir[-samp,])
print(max.col(dat))

quit()


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


trainPredictions <- matrix(data=0,nrow=nrow(Xtrain),ncol=1)

#sink("record.lis")

YY <- Xtrain[,1]	

pc.cr <- prcomp(x=Xtrain[,-1]) # , scale. =TRUE, retx=TRUE
print("PCA Done")
XtrainPCA <- predict(pc.cr, newdata=Xtrain[,-1])
print("PCA Predict train")
XtestPCA <- predict(pc.cr, newdata=Xtest)

#print(summary(pc.cr))
#print(summary(train))
#print(summary(Xtest))
print("Done PCA")
gc()	


#gctorture(on = TRUE)
#gctorture2(1, wait = 1, inhibit_release = FALSE)

bestScore = 0.0

for(j in 1:8) {
	#print(paste("Repeat: ", as.character(j)))

	for(i in 1:8) {
			
		gc()	
				
		features <- i*10
		neighbors = j  # 3 seems best...

		
		print(paste("Features: ", as.character(features)))
		print(paste("Neighbors:", as.character(neighbors)))
		# these are the real one for CV.

		XX <- XtrainPCA[,1:features] 
		
		#print(cbind( Freq=table(YY), Cumul=cumsum(table(YY)), Relative=prop.table(table(YY))))


		print(Sys.time())

		# from neighbors 2 - 8, 8 is best (0.9671).  check 8 - 16
		# 8 seems to be the best...
		
		# all using "cover_tree"
		# with pca-80 and knn-10, 0.9696
		# with PCA 80 , 1 = 0.9723, 2=0.9672, 3=9715, 4=9710, 5=9695, 6=9728, 7=9702, 8=9695, 9=9675, 10=9685
		# with PCA  40 features and neighbors 6 => 0.9741
		# with PCA  80 features and neighbors 6 => 0.9727
		# with PCA 40 and neighbors 4, => 0.9745, 6 was 0.9726 this run. depends of CV selection.
		# with PCA 40 1 => 9726, 2 =>  9690, 3 => 9737  (cv 10 was 97445), 4 => 9732, 5 => 9735, 6 => 9735, 7 => 9720, 8 => 9724, (5 cv)
		results <- knn.cv(train=XX, cl= YY, k = neighbors, algorithm="VR")
		# knn.cv uses leave-one-out validation.
		
		#print(results)
		print(Sys.time())

		currScore = as.double(sum(ifelse(results == YY, 1, 0)))/as.double(nrow(XtrainPCA))
		print (currScore)	

		if ( currScore > bestScore | currScore > 0.9740) {
			print("Found Best Score: Saving New Files")
			bestScore <- currScore
			filename <- paste("..\\predictions\\Target_Stack_", format(Sys.time(), '%Y%m%d%H%M%S'), "_", as.character(currScore), "_", "knn", ".csv", sep = "")
			write.table(trainPredictions, file=filename, row.names = FALSE, col.names = FALSE) #, row.names = FALSE, col.names = FALSE

			results <- knn(train=XX, test=XtestPCA[,1:features], cl= YY, k = neighbors, algorithm="VR")
			
			filename <- paste("..\\predictions\\Stack_", format(Sys.time(), '%Y%m%d%H%M%S'), "_", as.character(currScore), "_", "knn", ".csv", sep = "")
			write.table(results, file=filename, row.names = FALSE, col.names = FALSE)   

			
			cat("KNN", "\n", file = "..\\predictions\\RunLog.csv",sep="",append=TRUE) 
			cat("Features:", "\n", as.character(features) ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="",append=TRUE) 			
			cat("Neighbors:", "\n", as.character(neighbors) ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="",append=TRUE) 	
			cat("Score:", "\n", as.character(bestScore) ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="",append=TRUE) 	
			cat("DateTime:", "\n", format(Sys.time(), '%Y %m %d %H:%M:%S') ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="",append=TRUE) 	
			cat("DateTime:", "\n", format(Sys.time(), '%Y %m %d %H:%M:%S') ,"\r\n", file = "..\\predictions\\RunLog.csv",sep="",append=TRUE) 				
			cat("\n", file = "..\\predictions\\RunLog.csv",sep="",append=TRUE) 
		}
	 }
	 

	 
}
	 
	 





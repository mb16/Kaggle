require(nnet)
require(caret)
#library(mlbench)




X <- as.matrix(read.table("temp\\dataset_blend_trainX.txt",
                   header=FALSE,
                   sep= " "       
                   ))
Y <- as.matrix(read.table("temp\\pre_shuffled_target.csv",
                   header=FALSE,
                   sep= " "       
                   ))
Weights <- as.matrix(read.table("temp\\pre_shuffled_weights.csv",
                   header=FALSE,
                   sep= " "       
                   ))		   
Xtest <- as.matrix(read.table("temp\\dataset_blend_testX.txt",
                   header=FALSE,
                   sep= " "       
                   ))
				   


trainID <- as.matrix(read.table("temp\\pre_shuffled_train_id.csv",
                                header=TRUE,
                                sep= " "       
))
testID <- as.matrix(read.table("temp\\pre_shuffled_test_id.csv",
                               header=TRUE,
                               sep= " "       
))


print("Loaded Data")

dim(X)
dim(Y)

CVFolds = 10
folds <- createFolds(Y, k = CVFolds)


trainPredictions <- matrix(data=NA,nrow=nrow(X),ncol=1) #rep(NA, nrow(X)) 
testPredictions <- matrix(data=NA,nrow=nrow(Xtest),ncol=CVFolds)

cumScore <- 0


weighted_gini <- function(act,pred,weight) {
  df = data.frame(act=act, pred=pred, weight=weight)
  df <- df[order(df$pred, decreasing= TRUE),]
  df$random = cumsum(df$weight / sum(df$weight))
  df$cum_pos_found = cumsum(df$act * df$weight)
  sum(df$weight * (df$cum_pos_found / sum(df$act * df$weight) - df$random))
}

normalized_weighted_gini <- function(act,pred,weight) {
  weighted_gini(act,pred,weight) / weighted_gini(act,act,weight)
}



for(i in 1:CVFolds) {
        
	print(paste("Fold dim: ", as.character(i)))

	
	YY <- Y[-folds[[i]]]
	XX <- X[-folds[[i]],]	
	
	# note, gam is slightly better than the neural network.
	#lm.fit <- gam(YY ~ XX, family = gaussian, trace=TRUE) #family = binomial, for classification??
	lm.fit <- nnet(YY ~ XX, size = 2, rang = 0.1, decay = 5e-3, maxit = 300) # size = 7, rang = 0, decay = 0.1, maxit = 500
	#lm.fit <- lm(YY ~ XX) 

	YY <- Y[folds[[i]]]
	XX <- X[folds[[i]],]	
  WW <- Weights[folds[[i]]]
	#pred<-predict(lm.fit, newdata=data.frame(XX), type="response")
	pred <-predict(lm.fit,data.frame(XX), se.fit= TRUE) 
	
	trainPredictions[folds[[i]]]<- pred#$fit  # for LM
	#trainPredictions[folds[[i]]]<- pred
	
  
  currScore = normalized_weighted_gini(YY, pred, WW)
	#currScore = sum(Weights[folds[[i]]] * abs(pred*100.0 - Y[folds[[i]]])) / sum(Weights[folds[[i]]])
	cumScore <- cumScore + currScore
	print (currScore)
	
	#system.time(x <- matrix(data=NA,nrow=10000,ncol=10000))	
		
	XX = Xtest
	pred<-predict(lm.fit, newdata=data.frame(XX))

	
	#pred <-predict(lm.fit,data.frame(XX), se.fit= TRUE) 		
	testPredictions[,i]<- pred

		
 }



mat1 <- as.data.frame(rowMeans(testPredictions))
mat1$id <- testID
names(mat1)[names(mat1)=="rowMeans(testPredictions)"] <- "target"
write.table(mat1[c("id", "target")], file=paste("..\\submission\\Blend_", format(Sys.time(), "%Y%m%d%H%M%S"), "_", (cumScore/CVFolds), "_NNET.csv", sep = ""), row.names = FALSE, col.names = TRUE, sep = ",")   

dim(Y)
dim(trainPredictions)
dim(Weights)
#print(trainPredictions)


mat2 <- as.data.frame(trainPredictions)
mat2$id <- trainID
names(mat2)[names(mat2)=="V1"] <- "target"
write.table(mat2[c("id", "target")], file=paste("..\\submission\\Target_Blend_", format(Sys.time(), "%Y%m%d%H%M%S"), "_", (cumScore/CVFolds), "_NNET.csv", sep = ""), row.names = FALSE, col.names = TRUE, sep = ",") #, row.names = FALSE, col.names = FALSE

print(cumScore/CVFolds)
finalScore = normalized_weighted_gini(Y, trainPredictions, Weights)
print(finalScore)

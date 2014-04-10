#!/usr/bin/env python

from sklearn import svm
import math
import csv_io
import csv_io_np
from math import log
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest, chi2, f_classif
import scipy

import gc
import numpy as np
import datetime
import random
import os

from sklearn import preprocessing
	

def run_stack(SEED):

	model = "" 

	print "Running Stack."

	
	avg = 0
	NumFolds = 5 # 5 is good, but 10 yeilds a better mean since outliers are less significant. 

	targetX = csv_io.read_data("target.csv", skipFirstLine = False, split = ",")
	
	trainBase = csv_io.read_data("train1.csv", skipFirstLine = False, split = ",")	
	#test = csv_io_np.read_data("test1.csv", skipFirstLine = False, split = ",")

	trainBase = trainBase[0:5000]
	targetX = targetX[0:5000]
	
	train_saleID = csv_io.read_data("train_salesID.csv", skipFirstLine = False, split = ",")
	test_salesID = csv_io.read_data("test_salesID.csv", skipFirstLine = False, split = ",")
	

	predicted_list = []
	bootstrapLists = []


	clfs = [
		GradientBoostingRegressor(loss='ls', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=300, random_state=166, min_samples_leaf=1)		
	]		
	#GradientBoostingRegressor(loss='ls', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=300, random_state=166, min_samples_leaf=1)	
	#GradientBoostingRegressor(loss='lad', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=3000, random_state=166, min_samples_leaf=1),	
	#GradientBoostingRegressor(loss='huber', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=3000, random_state=166, min_samples_leaf=1),	
	

	
	print "Data size: ", len(trainBase) , 11573 # len(test)
	dataset_blend_train = np.zeros((len(trainBase), len(clfs)))
	#dataset_blend_test = np.zeros((len(test), len(clfs)))
	dataset_blend_test = np.zeros(11573, len(clfs))	
	
	#targetPre = target #[0:5000]
	#testScaled = test
	#trainScaled = trainBase #[0:5000]

	#targetPre = target #[0:5000]
	#testScaled = test
	#trainScaled = trainBase #[0:5000]
	
	
	print "Begin Training"

	lenTrainBase = len(trainBase)
	#lenTrainBase = len(trainBase[0:5000])


	lenTest = 11573
	#lenTest = len(test)

	
	
	gc.collect()
	
	for ExecutionIndex, clf in enumerate(clfs):
		print clf
		avg = 0
	
		predicted_list = []
			
		dataset_blend_test_set = np.zeros((lenTest, NumFolds))

		
		foldCount = 0

		
		#Stratified for classification...[trainBase[i][0] for i in range(len(trainBase))]
		Folds = cross_validation.KFold(lenTrainBase, k=NumFolds, indices=True)
			
		
		
		for train_index, test_index in Folds:

			target = [targetX[i] for i in train_index]
			train = [trainBase[i] for i in train_index]
			
			targetTest = [targetX[i] for i in test_index]	
			trainTest = [trainBase[i] for i in test_index]
			
			#target = [targetPre[i] for i in train_index]
			#train = [trainScaled[i] for i in train_index]
			
			#targetTest = [targetPre[i] for i in test_index]	
			#trainTest = [trainScaled[i] for i in test_index]	
	
			gc.collect()
			print
			print "Iteration: ", foldCount
			print "LEN: ", len(train), len(target)
			
			#print train[0]
			#print target[0]
			#return
			
			print "Start", datetime.datetime.now()
			clf.fit(train, target)
			prob = clf.predict(trainTest) 
			print "End  ", datetime.datetime.now()
			
			dataset_blend_train[test_index, ExecutionIndex] = prob

			gc.collect()

	
			probSum = 0
			weightSum = 0
			# totalOffByHalf = 0
			# totalPositive = 0
			# totalPositiveOffByHalf = 0
			# totalPositivePredictions = 0
			
			for i in range(0, len(prob)):
				probX = prob[i]
				probX = 31100.0
				print targetTest[i][0], probX
				probSum += math.pow(math.log10(targetTest[i][0]) - math.log10(probX), 2)
				
				#probSum += weights[test_index[i]][0] * math.fabs(targetTest[i] - probX)
				#weightSum += weights[test_index[i]][0] 
				
				
				#print "Weight", weights[test_index[i]][0], "Index: ",i, "Test_Index: ",test_index[i] , "Actual: ", targetTest[i], "Predicted: ", probX
				
				# log loss cal
				#probSum += int(targetTest[i])*log(probX)+(1-int(targetTest[i]))*log(1-probX)
				# if ( math.fabs(probX - int(targetTest[i])) > 0.5 ):
					# totalOffByHalf = totalOffByHalf + 1		
			
				# if ( int(targetTest[i]) == 1 ):
					# totalPositive = totalPositive + 1
				# if ( int(targetTest[i]) == 1 and probX < 0.5):
					# totalPositiveOffByHalf = totalPositiveOffByHalf + 1
				# if (probX > 0.5):
					# totalPositivePredictions = totalPositivePredictions + 1			
			
			# print
			# print "Stats:"
			# print "Total Off By > 0.5 ", totalOffByHalf
			# print "Total Positive ", totalPositive
			# print "Total Positive Off By Half ", totalPositiveOffByHalf
			# print "Total Positive Predictions ", totalPositivePredictions
			#print -probSum/len(prob)
			print "Score: ", math.sqrt(probSum/len(prob))
 
			avg += 	math.sqrt(probSum/len(prob))/NumFolds

			gc.collect()
			
			fo = open("test1.csv", "r")			
			predicted_probs = []
			
			for line in fo:
				line = line.strip().split(",")
				newRow = []		
				for item in line:
					newRow.append(float(item))
					
				predicted_probs.append(clf.predict(newRow))
				
			fo.close()
			
			#predicted_probs = clf.predict(testScaled) 	
			#predicted_list.append([x[1] for x in predicted_probs])	
			dataset_blend_test_set[:, foldCount] = predicted_probs #[0]
			gc.collect()
				
			foldCount = foldCount + 1
		
		dataset_blend_test[:,ExecutionIndex] = dataset_blend_test_set.mean(1)  
		
		#print "Saving NP"
		#np.savetxt('temp/dataset_blend_test_set.txt', dataset_blend_test_set)
		#np.savetxt('temp/dataset_blend_test_set.mean.txt', dataset_blend_test_set.mean(1) )
		#np.savetxt('temp/dataset_blend_test.txt', dataset_blend_test)
		#print "Done Saving NP"
		
		now = datetime.datetime.now()
		#print dataset_blend_test_set.mean(1) 
		csv_io.write_delimited_file_single("../predictions/Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_test_set.mean(1))
		
		csv_io.write_delimited_file_single("../predictions/Target_Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_train[:,ExecutionIndex] )		
		
		csv_io.write_delimited_file("../predictions/RunLog.csv", [now.strftime("%Y %m %d %H %M %S"), "AVG." , str(avg), str(clf), "Folds:", str(NumFolds), "Model", model, "", ""], filemode="a",delimiter=",")
		
		
		print "------------------------Average: ", avg

		#np.savetxt('temp/dataset_blend_train.txt', dataset_blend_train)

	return dataset_blend_train, dataset_blend_test
							
	
	
if __name__=="__main__":
	run_stack(448)
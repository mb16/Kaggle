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

import util
from collections import defaultdict
import pandas as pd

from sklearn import preprocessing

def get_date_dataframe(date_column):
    return pd.DataFrame({
        "SaleYear": [d.year for d in date_column],
        "SaleMonth": [d.month for d in date_column],
        "SaleDay": [d.day for d in date_column]
        }, index=date_column.index)
	

def run_stack(SEED):


	train, test = util.get_train_test_df()

	columns = set(train.columns)
	columns.remove("SalesID")
	columns.remove("SalePrice")
	columns.remove("saledate")

	
	train_fea = get_date_dataframe(train["saledate"])
	test_fea = get_date_dataframe(test["saledate"])

	for col in columns:
		types = set(type(x) for x in train[col])
		if str in types:
			s = set(x for x in train[col])
			str_to_categorical = defaultdict(lambda: -1, [(x[1], x[0]) for x in enumerate(s)])
			train_fea = train_fea.join(pd.DataFrame({col: [str_to_categorical[x] for x in train[col]]}, index=train.index))
			test_fea = test_fea.join(pd.DataFrame({col: [str_to_categorical[x] for x in test[col]]}, index=test.index))
		else:
			train_fea = train_fea.join(train[col])
			test_fea = test_fea.join(test[col])


	model = "" 
	print "Running Stack."

	
	avg = 0
	NumFolds = 5 # 5 is good, but 10 yeilds a better mean since outliers are less significant. 

	#targetX = csv_io.read_data("target.csv", skipFirstLine = False, split = ",")
	#trainBase = csv_io.read_data("train1.csv", skipFirstLine = False, split = ",")	
	#test = csv_io_np.read_data("test1.csv", skipFirstLine = False, split = ",")

	#trainBase = trainBase[0:5000]
	#targetX = targetX[0:5000]
	
	#train_saleID = csv_io.read_data("train_salesID.csv", skipFirstLine = False, split = ",")
	#test_salesID = csv_io.read_data("test_salesID.csv", skipFirstLine = False, split = ",")
	

	predicted_list = []
	bootstrapLists = []


	clfs = [
	
			GradientBoostingRegressor(loss='lad', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=3000, random_state=166, min_samples_leaf=1)
	]		
	#GradientBoostingRegressor(loss='ls', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=300, random_state=166, min_samples_leaf=1)	
	#GradientBoostingRegressor(loss='lad', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=3000, random_state=166, min_samples_leaf=1),	
	#GradientBoostingRegressor(loss='huber', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=3000, random_state=166, min_samples_leaf=1),	
	

	#train_fea, train["SalePrice"]
	print "Data size: ", len(train_fea) , len(test_fea)
	#dataset_blend_train = np.zeros((len(train_fea), len(clfs)))
	#dataset_blend_test = np.zeros((len(test), len(clfs)))
	dataset_blend_test = np.zeros((len(test_fea), len(clfs))) # np.zeros(len(train_fea), len(clfs))	
	dataset_blend_train = np.zeros((len(train_fea), len(clfs)))	

	
	print "Begin Training"

	lenTrainBase = 401125 # len(train_fea)



	lenTest = 11573 # len(test_fea)
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

			targetX = [train["SalePrice"][i] for i in train_index]
			trainX = [train_fea.ix[i] for i in train_index]
			
			targetTest = [train["SalePrice"][i] for i in test_index]	
			trainTest = [train_fea.ix[i] for i in test_index]
			

			gc.collect()
			print
			print "Iteration: ", foldCount
			print "LEN: ", len(trainX), len(targetX)
			
			#print trainX[0]
			#print target[0]
			#return
			
			print "Start", datetime.datetime.now()
			clf.fit(trainX, targetX)
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
				#print targetTest[i], probX
				
				if probX < 0: # some are comming out negative.
					probX = -probX			

				probSum += math.pow(math.log10(targetTest[i]) - math.log10(probX), 2)
				
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
			
		
			predicted_probs = []
			
			for i in range(0,lenTest):
				predicted_probs.append(clf.predict(test_fea.ix[i]))
				
			
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
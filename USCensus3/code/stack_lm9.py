#!/usr/bin/env python

from sklearn import svm
import csv_io
import math
from math import log
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import scipy

import gc

import datetime
import random

from sklearn import preprocessing
	
def run_stack(SEED):

	NumFeatures = 40
	Set = "" # "RFE" + "_"
	
	model = "Long-Lat KNN5 " + str(NumFeatures) + " Features, " + Set

	print "Running GB, RF, ET stack."

	trainBase = csv_io.read_data("PreProcessData/training_PreProcess9_" + Set + str(NumFeatures) + ".csv", skipFirstLine = False, split = "\t")
	test = csv_io.read_data("PreProcessData/test_PreProcess9_" + Set + str(NumFeatures) + ".csv", skipFirstLine = False, split = "\t")
	weights = csv_io.read_data("PreProcessData/Weights.csv", skipFirstLine = False)

	
	#random.seed(SEED)
	#random.shuffle(trainBase)
	
	avg = 0
	NumFolds = 5 # 5 is good, but 10 yeilds a better mean since outliers are less significant. (note, predictions are less reliable when using 10).


	predicted_list = []
	bootstrapLists = []

	# use this for quick runs.
	# note RF with 150 crashes on 30 features
	# GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
	# GradientBoostingRegressor(learn_rate=0.02, subsample=0.2, max_depth=8, n_estimators=125),
	# RandomForestRegressor(n_estimators=100, n_jobs=1),
	#RandomForestRegressor(n_estimators=75, n_jobs=1),
	# clfs = [ExtraTreesRegressor(n_estimators=50, min_density=0.2, n_jobs=1, bootstrap=True, random_state=1),
		# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=2**-5.5, kernel='rbf', probability=False, shrinking=True, tol=0.001, verbose=False)
		# ]	
	#knn 5 at 3.45
	#knn 15 at 3.31
	#knn 25 at 3.30
	#knn 40 at 3.31
	# KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=False, p=2),
	# KNeighborsRegressor(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=False, p=2),
	# KNeighborsRegressor(n_neighbors=25, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=False, p=2),
	# LinearRegression at 3.77
	# Ridge at 3.77
	# SGD 4.23
	#Gauss at 13
	# LinearRegression(fit_intercept=True, normalize=False, copy_X=True),
	# Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, tol=0.001),
	# SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, rho=0.84999999999999998, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.10000000000000001, p=None, seed=0, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False),
	# GaussianNB()
	# clfs = [KNeighborsRegressor(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=False, p=2),
		 # KNeighborsRegressor(n_neighbors=25, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=False, p=2),KNeighborsRegressor(n_neighbors=35, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=False, p=2)	
		# ]
		
	# GB, 125 est is minimum, score is bad below this, explore higher and other dimensions. ******************
	# clfs = [GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=100, random_state=166),
			# GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=8, n_estimators=100, random_state=166),
			# GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=10, n_estimators=100, random_state=166),
			# GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=200, random_state=166),
			# GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=8, n_estimators=200, random_state=166),GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=10, n_estimators=200, random_state=166)
			# ]	
			
	# about 1 hour run time, and 3.10 score.		
	#GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=400, random_state=166)
	# about 2 hours run time at 3.05
	#GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=8, n_estimators=400, random_state=166)
	# about 2 hours run time at 3.06
	#GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=800, random_state=166)
	# about 4 hours run time at 3.06
	#GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=8, n_estimators=800, random_state=166)	
	
	# 6/2000 on 40 features is 2.97
	# GradientBoostingRegressor(max_features=40, learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=3000, random_state=166, min_samples_leaf=1)
	clfs = [
		GradientBoostingRegressor(loss='ls', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=2000, random_state=166, min_samples_leaf=1),	
		GradientBoostingRegressor(loss='lad', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=3000, random_state=166, min_samples_leaf=1),	
		GradientBoostingRegressor(loss='huber', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=3000, random_state=166, min_samples_leaf=1),	
	]		
	
	# GradientBoostingRegressor(loss='ld', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=2000, random_state=166, min_samples_leaf=1)	
	# GradientBoostingRegressor(loss='lad', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=2000, random_state=166, min_samples_leaf=1)	
	# GradientBoostingRegressor(loss='huber', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=2000, random_state=166, min_samples_leaf=1)	

	
	
	print "Data size: ", len(trainBase), len(test)
	dataset_blend_train = np.zeros((len(trainBase), len(clfs)))
	dataset_blend_test = np.zeros((len(test), len(clfs)))
	

	trainNew = []
	trainTestNew = []
	testNew = []
	trainNewSelect = []
	trainTestNewSelect = []
	testNewSelect = []
	
	print "Scaling"
	targetPre = [x[0] for x in trainBase]
	trainPre = [x[1:] for x in trainBase]
	testPre = [x[0:] for x in test]
	#print trainPre[0]
	scaler = preprocessing.Scaler().fit(trainPre)
	trainScaled = scaler.transform(trainPre)
	testScaled = scaler.transform(testPre)	

	#print scaler.mean_
	#print scaler.std_
	print "Begin Training"
	
	lenTrainBase = len(trainBase)
	trainBase = []
	
	lenTest = len(test)
	test = []
	
	trainPre = []
	testPre = []
	
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

			#trainBaseTemp = [trainBase[i] for i in train_index]
			#target = [x[0] for x in trainBaseTemp]
			#train = [x[1:] for x in trainBaseTemp]
	
			#testBaseTemp = [trainBase[i] for i in test_index]
			#targetTest = [x[0] for x in testBaseTemp]
			#trainTest = [x[1:] for x in testBaseTemp]
		
			#test = [x[0:] for x in test]
	
			target = [targetPre[i] for i in train_index]
			train = [trainScaled[i] for i in train_index]
			
			targetTest = [targetPre[i] for i in test_index]	
			trainTest = [trainScaled[i] for i in test_index]	
	
			print
			print "Iteration: ", foldCount
			print "LEN: ", len(train), len(target)
			
			clf.fit(train, target)
			prob = clf.predict(trainTest) 
			
			dataset_blend_train[test_index, ExecutionIndex] = prob



	
			probSum = 0
			weightSum = 0
			# totalOffByHalf = 0
			# totalPositive = 0
			# totalPositiveOffByHalf = 0
			# totalPositivePredictions = 0
			
			for i in range(0, len(prob)):
				probX = prob[i]

				probSum += weights[test_index[i]][0] * math.fabs(targetTest[i] - probX)
				weightSum += weights[test_index[i]][0] 
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
			print "Score: ", probSum/weightSum
 
			avg += 	(probSum/weightSum)/NumFolds

			predicted_probs = clf.predict(testScaled) 	
			#predicted_list.append([x[1] for x in predicted_probs])	
			dataset_blend_test_set[:, foldCount] = predicted_probs #[0]
		
				
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
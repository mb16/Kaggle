#!/usr/bin/env python

from sklearn import svm
import csv_io_np
import csv_io
import math
from math import log
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
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


	model = "base"

	trainBase = csv_io_np.read_data("PreProcessData/train.csv", skipFirstLine = True, split = ",")
	test = csv_io_np.read_data("PreProcessData/test.csv", skipFirstLine = True, split = ",")

	print "Data Read Complete"
	
	avg = 0
	NumFolds = 5 


	predicted_list = []
	bootstrapLists = []

	# 100 producted 94% 
	# 1000 did not finish in about 5+ hours...
	# 300 about 5 hours, .9691 on first CF
	# learn_rate=0.01, n_estimators=300, subsample=1.0, min_samples_split=30, 0.9386
	#		GradientBoostingClassifier(loss='deviance', learn_rate=0.1, n_estimators=300, subsample=1.0, min_samples_split=30, min_samples_leaf=1, max_depth=5, init=None, random_state=None, max_features=None)
	
	# Leader board of 98443, for 20th place.
	#SVC(C=10**6, kernel='rbf', degree=3, gamma=10**-6.35, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False)

	clfs = [
		SVC(C=10**6, kernel='rbf', degree=3, gamma=10**-6.35, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False)
		]		
	
	
	
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
	
	# image best restuls [-1, 1]
	#preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
	
	#print trainPre[0]
	#scaler = preprocessing.Scaler().fit(trainPre)
	#trainScaled = scaler.transform(trainPre)
	#testScaled = scaler.transform(testPre)	
	trainScaled = trainPre
	testScaled = testPre
	
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
			print "LEN: ", len(train), len(train[0]), len(target), len(trainTest), len(trainTest[0])
			
			print datetime.datetime.now()
			clf.fit(train, target)
			print datetime.datetime.now()
			prob = clf.predict(trainTest) 
			
			dataset_blend_train[test_index, ExecutionIndex] = prob



	
			probSum = 0.0
			count = 0.0

			
			for i in range(0, len(prob)):
				probX = prob[i]#[1]
				#print probX, targetTest[i]
				if ( targetTest[i] == probX ) :
					probSum += 1.0
				count = count + 1.0
		
			print "Sum: ", probSum, count
			print "Score: ", probSum/count
 
			avg += 	(probSum/count)/NumFolds

			
			#predicted_probs = clf.predict(testScaled) 	
			######predicted_list.append([x[1] for x in predicted_probs])	
			#dataset_blend_test_set[:, foldCount] = predicted_probs #[0]
		
				
			foldCount = foldCount + 1
		
		print "Final Train", datetime.datetime.now()
		clf.fit(trainScaled, targetPre)  # must to this for multiclass classification...
		print "Final Predict", datetime.datetime.now()
		predicted_probs = clf.predict(testScaled) 	
		print "Writing Data", datetime.datetime.now()
		#dataset_blend_test[:,ExecutionIndex] = dataset_blend_test_set.mean(1)  
		

		
		now = datetime.datetime.now()

		#csv_io.write_delimited_file_single("../predictions/Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_test_set.mean(1))
		csv_io.write_delimited_file_single("../predictions/Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", predicted_probs) # for multiclass
		
		csv_io.write_delimited_file_single("../predictions/Target_Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_train[:,ExecutionIndex] )		
		
		csv_io.write_delimited_file("../predictions/RunLog.csv", [now.strftime("%Y %m %d %H %M %S"), "AVG." , str(avg), str(clf), "Folds:", str(NumFolds), "Model", model, "", ""], filemode="a",delimiter=",")
		
		
		print "------------------------Average: ", avg



	return dataset_blend_train, dataset_blend_test
							
	
	
if __name__=="__main__":
	run_stack(448)
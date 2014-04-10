#!/usr/bin/env python

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
import csv_io
import math

from math import log
import string

import stack

import numpy as np
import datetime

import random
import os

def KNNWeight(arr):
	#print "myFunc: ", arr
	arr1 = arr[0]
	
	newArr = []
	for index, data in enumerate(arr1):
		if index == 0:
			newArr.append(0.0)
		else:
			#newArr.append(1.0/data)
			newArr.append(1.0)
	#print newArr
	return [newArr]
	


def Blend():

	lossThreshold = 4.0  # best seems to be about 4.0
	model = "Long-Lat KNN5"

	#used only for targets values.
	trainBase = csv_io.read_data("PreProcessData/training_PreProcess4.csv", skipFirstLine = False, split = "\t")
	test = csv_io.read_data("PreProcessData/test_PreProcess4.csv", False)
	weights = csv_io.read_data("PreProcessData/Weights.csv", skipFirstLine = False)

	target = [x[0] for x in trainBase]
	
	
	
	
	stackFiles = []
	for filename in os.listdir("../predictions"):
		parts = filename.split("_")
		if ( filename[0:5] == "Stack" and float(parts[2]) < lossThreshold):

			stackFiles.append(filename)
	
	
	dataset_blend_train = np.zeros((len(trainBase), len(stackFiles)))
	dataset_blend_test = np.zeros((len(test), len(stackFiles)))
	
	print "Loading Data"
	for fileNum, file in enumerate(stackFiles):
		print file
		trn = csv_io.read_data("../predictions/Target_" + file, split="," ,skipFirstLine = False)
		for row, datum in enumerate(trn):
			dataset_blend_train[row, fileNum] = datum[0]
		
		tst = csv_io.read_data("../predictions/" + file, split="," ,skipFirstLine = False)
		for row, datum in enumerate(tst):
			dataset_blend_test[row, fileNum] = datum[0]

	np.savetxt('temp/dataset_blend_trainX.txt', dataset_blend_train)
	np.savetxt('temp/dataset_blend_testX.txt', dataset_blend_test)
	print "Num file processed: ", len(stackFiles), "Threshold: ", lossThreshold

	
	# linear 3.15 -> 3.42
	# RF 1.2 -> 3.5
	# GB (125) 3.15
	
	
	
	
	
	print "Starting Blend"
	#GB 400 is 3.11
	#GB 400 max_depth=14 is 2.82  greater depth is better.
	# GB seems to overfit. scores drop for 100 estimators = 3.33 with linear on same code, 3.27
	# might try smaller numbers in gb than 20 depth and 100 est with prevent overfitting.
	
	# clfs = [
		# GradientBoostingRegressor(learn_rate=0.05, subsample=0.2, max_depth=20, n_estimators=400, random_state=551),
		# GradientBoostingRegressor(learn_rate=0.05, subsample=0.2, max_depth=30, n_estimators=400, random_state=551),
		# GradientBoostingRegressor(learn_rate=0.05, subsample=0.2, max_depth=40, n_estimators=400, random_state=551),
		# GradientBoostingRegressor(learn_rate=0.05, subsample=0.2, max_depth=80, n_estimators=400, random_state=551),
		# GradientBoostingRegressor(learn_rate=0.05, subsample=0.2, max_depth=20, n_estimators=800, random_state=551),
		# GradientBoostingRegressor(learn_rate=0.05, subsample=0.2, max_depth=30, n_estimators=800, random_state=551),
		# GradientBoostingRegressor(learn_rate=0.05, subsample=0.2, max_depth=40, n_estimators=800, random_state=551),
		# GradientBoostingRegressor(learn_rate=0.05, subsample=0.2, max_depth=80, n_estimators=800, random_state=551)
	
		# ]
		
	clfs = [Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, tol=0.001)
	]	
	
	# this returned 2.95 when linear returned 3.06,  need to check for overfitting.
	#KNeighborsRegressor(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=False, p=2)
	
	# linear 3.06, lasso is 3.06
	#Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute='auto', copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False)
	
	#linear 3.06, ridge 3.05
	#Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, tol=0.001)
	
	#linear 3.06, SVC 2.77, not sure if overfitting, need to submit to test**************
	#SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False)
	
	
	# clfs = [LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
	# ]	
		
	#LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
	# use for classification probablilities
	# clfs = [LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None),
			# LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.5, fit_intercept=True, intercept_scaling=1, class_weight=None),
			# LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.1, fit_intercept=True, intercept_scaling=1, class_weight=None),
			# LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None),
			# LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.5, fit_intercept=True, intercept_scaling=1, class_weight=None),
			# LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.1, fit_intercept=True, intercept_scaling=1, class_weight=None)]
	

	dataset_blend_test_set = np.zeros((len(test), len(clfs)))
	
	avgScore = 0.0
	for ExecutionIndex, clf in enumerate(clfs):
		print clf
		clf.fit(dataset_blend_train, target)
		submission = clf.predict(dataset_blend_test)
		
		submission = ["%f" % x for x in submission]
		now = datetime.datetime.now()

		

		
		# attempt to score the training set to predict score for blend...
		probSum = 0.0
		weightSum = 0
		
		trainPrediction = clf.predict(dataset_blend_train)
		for i in range(0, len(trainPrediction)):
			probX = trainPrediction[i]
			

			probSum += weights[i][0] * math.fabs(target[i] - probX)
			weightSum += weights[i][0]
			#probSum += int(target[i])*log(probX)+(1-int(target[i]))*log(1-probX)
			 
		print "Train Score: ", (probSum/weightSum)
		avgScore += (probSum/weightSum)
	
		csv_io.write_delimited_file("../blend/BlendSingle" + now.strftime("%Y%m%d%H%M%S") + "_" + str(probSum/weightSum)+ "_" + str(clf)[:12] + ".csv", submission)	
	
		csv_io.write_delimited_file("../blend/RunLog.csv", [now.strftime("%Y %m %d %H %M %S"), str(avgScore/len(clfs)), str(clf), "1", model, "", "", ", ".join(stackFiles)], filemode="a",delimiter=",")
	
		dataset_blend_test_set[:, ExecutionIndex] = submission
	
	
	print "Final Score: ", str(avgScore/len(clfs))
	
	csv_io.write_delimited_file_single("../blend/FinalBlend_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avgScore/len(clfs)) + ".csv", dataset_blend_test_set.mean(1))	


	
											
if __name__=="__main__":
	Blend()
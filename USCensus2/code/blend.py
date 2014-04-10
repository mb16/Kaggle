#!/usr/bin/env python

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import csv_io
import math

from math import log
import string

import stack

import numpy as np
import datetime

import random

def Blend():

	trainBase = csv_io.read_data("PreProcessData/training_PreProcess4.csv", skipFirstLine = False, split = "\t")
	weights = csv_io.read_data("PreProcessData/Weights.csv", skipFirstLine = False)
		
	SEED = 448
	#random.seed(SEED)
	#random.shuffle(trainBase)
	
	target = [x[0] for x in trainBase]
	
	dataset_blend_train, dataset_blend_test = stack.run_stack(SEED)
	clfs = [LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
		]
	
	
	# clfs = [LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None),
			# LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.5, fit_intercept=True, intercept_scaling=1, class_weight=None),
			# LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.1, fit_intercept=True, intercept_scaling=1, class_weight=None),
			# LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None),
			# LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.5, fit_intercept=True, intercept_scaling=1, class_weight=None),
			# LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.1, fit_intercept=True, intercept_scaling=1, class_weight=None)]
	
	test = csv_io.read_data("PreProcessData/test_PreProcess4.csv", False)
	dataset_blend_test_set = np.zeros((len(test), len(clfs)))
	
	for ExecutionIndex, clf in enumerate(clfs):

		clf.fit(dataset_blend_train, target)
		submission = clf.predict(dataset_blend_test)
		
		submission = ["%f" % x for x in submission]
		now = datetime.datetime.now()
		csv_io.write_delimited_file("../Submissions/BlendSingle" + now.strftime("%Y%m%d%H%M%S") + ".csv", submission)	
		

		
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
	
		dataset_blend_test_set[:, ExecutionIndex] = submission
	
	
	
	csv_io.write_delimited_file_single("../Submissions/FinalBlend_" + now.strftime("%Y%m%d%H%M%S") + ".csv", dataset_blend_test_set.mean(1))	

											
if __name__=="__main__":
	Blend()
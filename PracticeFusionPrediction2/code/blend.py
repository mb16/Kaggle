#!/usr/bin/env python

from sklearn.linear_model import LogisticRegression
import csv_io
import math

from math import log
import string

import stack

import numpy as np
import datetime

import random

def Blend():

	trainBase = csv_io.read_data("PreProcessData/training_PreProcess4.csv", False)
	
	SEED = 448
	random.seed(SEED)
	random.shuffle(trainBase)
	
	target = [x[0] for x in trainBase]
	
	dataset_blend_train, dataset_blend_test = stack.run_stack(SEED)

	clfs = [LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None),
			LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.5, fit_intercept=True, intercept_scaling=1, class_weight=None),
			LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.1, fit_intercept=True, intercept_scaling=1, class_weight=None),
			LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None),
			LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.5, fit_intercept=True, intercept_scaling=1, class_weight=None),
			LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.1, fit_intercept=True, intercept_scaling=1, class_weight=None)]
	
	test = csv_io.read_data("PreProcessData/test_PreProcess4.csv", False)
	dataset_blend_test_set = np.zeros((len(test), len(clfs)))
	
	for ExecutionIndex, clf in enumerate(clfs):

		clf.fit(dataset_blend_train, target)
		submission = clf.predict_proba(dataset_blend_test)[:,1]
		
		submission = ["%f" % x for x in submission]
		now = datetime.datetime.now()
		csv_io.write_delimited_file_GUID("../Submissions/stack" + now.strftime("%Y%m%d%H%M%S") + ".csv", "PreProcessData/test_PatientGuid.csv", submission)	
		

		
		# attempt to score the training set to predict score for blend...
		probSum = 0.0
		trainPrediction = clf.predict_proba(dataset_blend_train)[:,1]
		for i in range(0, len(trainPrediction)):
			probX = trainPrediction[i]
			if ( probX > 0.999):
				probX = 0.999;		
			if ( probX < 0.001):
				probX = 0.001;

			probSum += int(target[i])*log(probX)+(1-int(target[i]))*log(1-probX)
			 
		print "Train Score: ", (-probSum/len(trainPrediction))
	
		dataset_blend_test_set[:, ExecutionIndex] = submission
	
	
	
	csv_io.write_delimited_file_GUID_numpy("../Submissions/FinalBlend_" + now.strftime("%Y%m%d%H%M%S") + ".csv", "PreProcessData/test_PatientGuid.csv", dataset_blend_test_set.mean(1))	

											
if __name__=="__main__":
	Blend()
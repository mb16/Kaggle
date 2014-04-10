#!/usr/bin/env python

from sklearn.linear_model import LogisticRegression
import csv_io
import math

from math import log
import string

import stack_gb

import numpy as np
import datetime

def Blend():

	trainBase = csv_io.read_data("PreProcessData/training_PreProcess2.csv", False)
	target = [x[0] for x in trainBase]
	
	dataset_blend_train, dataset_blend_test = stack_gb.run_stack()

	clf = LogisticRegression()
	clf.fit(dataset_blend_train, target)
	submission = clf.predict_proba(dataset_blend_test)[:,1]
	
 	submission = ["%f" % x for x in submission]
	now = datetime.datetime.now()
	csv_io.write_delimited_file_GUID("../Submissions/stack_" + now.strftime("%Y%m%d%H%M") + ".csv", "PreProcessData/test_PatientGuid.csv", submission)	
	

	
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
	
	
	
	var = raw_input("Enter to terminate.")	
											
if __name__=="__main__":
	Blend()
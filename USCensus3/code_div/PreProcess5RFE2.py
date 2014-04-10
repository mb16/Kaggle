#!/usr/bin/env python


import string

from sklearn import svm
import csv_io
import math
from math import log
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor,GradientBoostingRegressor
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import scipy
from sklearn import preprocessing
import datetime
from sklearn.linear_model import Lasso, LinearRegression
import operator

import gc

import shutil

def toFloat(str):
	return float(str)


def PreProcess5():

	#note, 275 represents too much data, and the scaler fails with an exception.

	trainBase = csv_io.read_data("PreProcessData/training_PreProcess5_250.csv", skipFirstLine = False, split = "\t")
	test = csv_io.read_data("PreProcessData/test_PreProcess5_250.csv", skipFirstLine = False, split = "\t")
	#shutil.copy2("PreProcessData/DataClassList5.csv", "PreProcessData/DataClassList6.csv")	
	
	target = [x[0] for x in trainBase]
	train = [x[1:] for x in trainBase]
	
	DataClassList = csv_io.read_data("PreProcessData/DataClassList_Importances_250.csv", False)
	
	print "Data len: ", len(train[0])
	print "DataClassList len: ", len(DataClassList)
	#return
	
	# this seems about optimal, but has not been tuned on latest improvements.
	NumFeatures = 40
	# NOTE going from 30 to 20 features on KNN5 set has almost no effect.  Down to 15 is significant loss.
	# for GBM at 6 and 400 30 is 3.01 and 30 3.05.
	
	print "Scaling"
	targetPre = [x[0] for x in trainBase][0:10000]
	print "Scaling1"
	trainPre = [x[1:] for x in trainBase][0:10000]
	#testPre = [x[0:] for x in test]
	print "Scaling2"
	scaler = preprocessing.Scaler().fit(trainPre)
	print "Scaling3"
	trainScaled = scaler.transform(trainPre)
	#testScaled = scaler.transform(testPre)	
	

	#clf = RandomForestRegressor(n_estimators=25, n_jobs=1,compute_importances=True) 

	#gc.collect()
	

	print "Prep Classes"

	# prep for usage below...
	DataClassListTemp = []
	for DataIndex, DataClass in enumerate(DataClassList):
		DataClassListTemp.append([DataClass[0], 0])
	
	DataClassList = DataClassListTemp
	
	
	reduceBy = 5
	totalFeatures = len(trainPre[0])
	
	trainNew = []
	testNew = []
	
	print "Processing"
	while ( totalFeatures > NumFeatures):
	
		if ( totalFeatures - NumFeatures < 40 ) :
			reduceBy = 3
		if ( totalFeatures - NumFeatures < 20 ) :
			reduceBy = 2
		if ( totalFeatures - NumFeatures < 10 ) :
			reduceBy = 1
			
	
		if ( totalFeatures - NumFeatures < reduceBy):
			reduceBy = totalFeatures - NumFeatures
			print "Reduce Features: ", reduceBy
		
		print "Training"
		clf = GradientBoostingRegressor(loss='ls', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=400, random_state=166, min_samples_leaf=30)
		clf.fit(trainScaled, targetPre)
					
		print "Computing Importances"
		importances = clf.feature_importances_
		#print importances
		importancesSorted = sorted(importances, reverse=True)
		#print importancesSorted
		threshold = importancesSorted[len(importancesSorted) - reduceBy]
		print threshold
		#trainScaled = clf.transform(trainScaled, threshold) # only exists in RF

		
		trainScaledNew = []
		for row in trainScaled:
			newRow = []
			for impIndex, importance in enumerate(importances):
				if ( importance > threshold ):	
					newRow.append(row[impIndex])
			trainScaledNew.append(newRow)
			
		trainScaled = trainScaledNew
		
		print "Cols:", len(trainScaled)
		print "Rows:", len(trainScaled[0])
		
		totalFeatures = totalFeatures - reduceBy
		print "Total Features:", totalFeatures
	
	
		trainNew = []
		testNew = []

		for row in train:
			newRow = []
			for impIndex, importance in enumerate(importances):
				if ( importance > threshold ):	
					newRow.append(row[impIndex])
			trainNew.append(newRow)	
			
		train = trainNew	
			
		for row in test:
			newRow = []
			for impIndex, importance in enumerate(importances):
				if ( importance > threshold ) :
					newRow.append(row[impIndex])
			testNew.append(newRow)	
			
		test = testNew
	
		print "Train Cols:", len(train)
		print "Train Rows:", len(train[0])
	
		print "Test Cols:", len(test)
		print "Test Rows:", len(test[0])
	
	
		DataClassListNew = []
		for Index, importance in enumerate(importances):
				if ( importance > threshold ) :
					print DataClassList[Index][0], importance
					DataClassListNew.append([DataClassList[Index][0], importance])
		
			
		DataClassList = DataClassListNew
	
		print "Data Transform Complete"
	
	
	# final steps, save data classes in new set
		
	csv_io.write_delimited_file("PreProcessData/DataClassList_Importances_RFE2_" + str(NumFeatures) + ".csv", DataClassListNew)
	
	DataClassListNew_temp = sorted(DataClassListNew, key=operator.itemgetter(1), reverse=True)   
	csv_io.write_delimited_file("PreProcessData/DataClassList_Importances_RFE2_sorted_" + str(NumFeatures) + ".csv", DataClassListNew_temp)

		
	
	# prepend the target on each row.
	trainFinal = []

	rowIndex = 0
	for row in train:
		newRow = []
		for Index, val in enumerate(row):
			if ( Index == 0):
				newRow.append(target[rowIndex])
			newRow.append(val)
		trainFinal.append(newRow)	
		rowIndex += 1
			
							
	csv_io.write_delimited_file("PreProcessData/training_PreProcess6_RFE2_" + str(NumFeatures) + ".csv", trainFinal, delimiter="\t")		
	csv_io.write_delimited_file("PreProcessData/test_PreProcess6_RFE2_" + str(NumFeatures) + ".csv", testNew, delimiter="\t")
	
	
								
if __name__=="__main__":
	PreProcess5()
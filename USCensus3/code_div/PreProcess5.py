#!/usr/bin/env python


import string

from sklearn import svm
import csv_io
import math
from math import log
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_selection import RFE
import scipy
from sklearn import preprocessing
import datetime
from sklearn.linear_model import Lasso, LinearRegression
import operator

import gc

import shutil

def toFloat(str):
	return float(str)


def PreProcess5(N_Features):

	N_Features = 250

	trainBase = csv_io.read_data("PreProcessData/training_PreProcess4.csv", skipFirstLine = False, split = "\t")
	
	shutil.copy2("PreProcessData/DataClassList4.csv", "PreProcessData/DataClassList5.csv")	
	
	target = [x[0] for x in trainBase]
	train = [x[1:] for x in trainBase]
	
	DataClassList = csv_io.read_data("PreProcessData/DataClassList5.csv", False)
	
	print "Data len: ", len(train[0])
	print "DataClassList len: ", len(DataClassList)
	#return
	
	# this seems about optimal, but has not been tuned on latest improvements.
	NumFeatures = N_Features
	# NOTE going from 30 to 20 features on KNN5 set has almost no effect.  Down to 15 is significant loss.
	# for GBM at 6 and 400 30 is 3.01 and 30 3.05.
	
	print "Scaling"
	term = 5000 #  scaler has memory errors between 5000 and 10000
	#term = len(trainBase)
	targetPre = [x[0] for x in trainBase][0:term]
	trainPre = [x[1:] for x in trainBase][0:term]
	#testPre = [x[0:] for x in test][0:term]
	targetPre = target[0:term]
	#print trainPre[term - 1]
	
	# generates python exceptions...
	#scaler = preprocessing.Scaler().fit(trainPre)
	#trainScaled = scaler.transform(trainPre)
	
	#testScaled = scaler.transform(testPre)	

	
	clf = GradientBoostingRegressor(loss='ls', learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=400, random_state=166, min_samples_leaf=30)
	#clf = RandomForestRegressor(n_estimators=15, n_jobs=1,compute_importances=True) 
	
	print "Training"

	clf.fit(trainPre, targetPre)
	#clf.fit(trainScaled, targetPre)
		
	trainNew = []
	testNew = []

		
	print "Computing Importances"
	importances = clf.feature_importances_

	
	

	DataClassListNew = []
	for DataIndex, DataClass in enumerate(DataClassList):
		print DataClass[0], importances[DataIndex];
		DataClassListNew.append([DataClass[0], importances[DataIndex]])
		
	csv_io.write_delimited_file("PreProcessData/DataClassList_Importances_" + str(NumFeatures) + ".csv", DataClassListNew)
	
	DataClassListNew_temp = sorted(DataClassListNew, key=operator.itemgetter(1), reverse=True)  
	csv_io.write_delimited_file("PreProcessData/DataClassList_Importances_sorted_" + str(NumFeatures) + ".csv", DataClassListNew_temp)

	
	
	importancesTemp = sorted(importances, reverse=True)
	print len(importancesTemp), "importances"
				
	if ( len(importancesTemp) > NumFeatures):
		threshold = importancesTemp[NumFeatures]

		print "Importance threshold: ", threshold

		# ----- Comment out if getting a mem error loadint test data....
		rowIndex = 0
		for row in train:
			newRow = []
			
			for impIndex, importance in enumerate(importances):
				if ( impIndex == 0):
					newRow.append(target[rowIndex])
				if ( importance > threshold ):	
					newRow.append(row[impIndex])
			trainNew.append(newRow)	
			if ( rowIndex == 0 ) :
				print newRow
			rowIndex += 1
			
		print "Saving Train"	
		csv_io.write_delimited_file("PreProcessData/training_PreProcess5_" + str(NumFeatures) + ".csv", trainNew, delimiter="\t")	
		# -------
		
		train = []
		trainNew = []
		trainBase = []
		
		gc.collect()
		
		print "Load Test"
		test = csv_io.read_data("PreProcessData/test_PreProcess4.csv", skipFirstLine = False, split = "\t")	
			
		rowIndex = 0	
		for row in test:
			newRow = []
			for impIndex, importance in enumerate(importances):
				if ( importance > threshold ) :
					newRow.append(row[impIndex])
			testNew.append(newRow)
			
			if ( rowIndex == 0 ) :			
				print newRow
			
			rowIndex += 1


	print "Save Test"		
		
	csv_io.write_delimited_file("PreProcessData/test_PreProcess5_" + str(NumFeatures) + ".csv", testNew, delimiter="\t")
	
	
								
if __name__=="__main__":
	PreProcess5(40)
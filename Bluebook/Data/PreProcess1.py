#!/usr/bin/env python


import string

from sklearn import svm
import csv_io
import csv_io_np
import math
from math import log
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_selection import RFE
import scipy
from sklearn import preprocessing
import datetime
from sklearn.linear_model import Lasso, LinearRegression
import operator

import shutil

def toFloat(str):
	return float(str)


def PreProcess1(N_Features):

	trainBase = csv_io_np.read_data("train1.csv", skipFirstLine = False, split = ",")	
	test = csv_io_np.read_data("test1.csv", skipFirstLine = False, split = ",")

	target = csv_io.read_data("target.csv", skipFirstLine = False, split = ",")
	
	train = trainBase
	
	NumFeatures = N_Features
	

	#clf = RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='gini',compute_importances=True)
	clf = RandomForestRegressor(n_estimators=25, n_jobs=1,compute_importances=True) 
	#clf = ExtraTreesClassifier(n_estimators=150, min_density=0.02, n_jobs=1, criterion='gini', bootstrap=True,compute_importances=True)
	
	print "Training"
	
	clf.fit(train, target)

	print "Computing Importances"
	print clf.feature_importances_
	return
	newTest = clf.transform(test, 0.1)
	newTrain = clf.transform(train, 0.1)		
		
	csv_io.write_delimited_file("train2.csv", newTrain, delimiter=",")		
	csv_io.write_delimited_file("test2.csv", newTest, delimiter=",")


	return
		
		
		
		
	trainNew = []
	testNew = []


	DataClassListNew = []
	for DataIndex, DataClass in enumerate(DataClassList):
		print DataClass[0], importances[DataIndex];
		DataClassListNew.append([DataClass[0], importances[DataIndex]])
		
	csv_io.write_delimited_file("PreProcessData/DataClassList_Importances_Base_" + str(NumFeatures) + ".csv", DataClassListNew)
	
	DataClassListNew_temp = sorted(DataClassListNew, key=operator.itemgetter(1), reverse=True)  
	csv_io.write_delimited_file("PreProcessData/DataClassList_Importances_sorted_Base_" + str(NumFeatures) + ".csv", DataClassListNew_temp)

	
	
	importancesTemp = sorted(importances, reverse=True)
	print len(importancesTemp), "importances"
	
				
	if ( len(importancesTemp) > NumFeatures):
		threshold = importancesTemp[NumFeatures]

		print "Importance threshold: ", threshold

		rowIndex = 0
		for row in train:
			newRow = []
			for impIndex, importance in enumerate(importances):
				if ( impIndex == 0):
					newRow.append(target[rowIndex])
				if ( importance > threshold ):	
					newRow.append(row[impIndex])
			trainNew.append(newRow)	
			rowIndex += 1
			
			
		for row in test:
			newRow = []
			for impIndex, importance in enumerate(importances):
				if ( importance > threshold ) :
					newRow.append(row[impIndex])
			testNew.append(newRow)	
				
	csv_io.write_delimited_file("PreProcessData/training_PreProcess4_Base_" + str(NumFeatures) + ".csv", trainNew, delimiter="\t")		
	csv_io.write_delimited_file("PreProcessData/test_PreProcess4_Base_" + str(NumFeatures) + ".csv", testNew, delimiter="\t")
	
	
								
if __name__=="__main__":
	PreProcess1(200)
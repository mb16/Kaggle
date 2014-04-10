#!/usr/bin/env python


import string

from sklearn import svm
import csv_io
import math
from math import log
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import scipy

import datetime

import operator

def toFloat(str):
	return float(str)


def PreProcess4():

	trainBase = csv_io.read_data("PreProcessData/training_PreProcess3.csv", False)
	test = csv_io.read_data("PreProcessData/test_PreProcess3.csv", False)

	target = [x[0] for x in trainBase]
	train = [x[1:] for x in trainBase]
	
	# this seems about optimal, but has not been tuned on latest improvements.
	NumFeatures = 200
	
	#clf = RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='gini',compute_importances=True)
	clf = RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='entropy',compute_importances=True) # this is good, but ExtraTrees is better on Bio Prediction.
	#clf = ExtraTreesClassifier(n_estimators=150, min_density=0.02, n_jobs=1, criterion='gini', bootstrap=True,compute_importances=True)
	
	print "Training"
	clf.fit(train, target)
		
	trainNew = []
	testNew = []

		
	print "Computing Importances"
	importances = clf.feature_importances_

	
	
	
	DataClassList = csv_io.read_data("PreProcessData/DataClassList.csv", False)
	DataClassListNew = []
	for DataIndex, DataClass in enumerate(DataClassList):
		print DataClass[0], importances[DataIndex];
		DataClassListNew.append([DataClass[0], importances[DataIndex]])
	csv_io.write_delimited_file("PreProcessData/DataClassList_Importances.csv", DataClassListNew)
	
	DataClassListNew_temp = sorted(DataClassListNew, key=operator.itemgetter(1), reverse=True)  
	csv_io.write_delimited_file("PreProcessData/DataClassList_Importances_sorted.csv", DataClassListNew_temp)

	
	
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
				if ( importance > threshold ) :	
					newRow.append(row[impIndex])
			trainNew.append(newRow)	
			rowIndex += 1
			
			
		for row in test:
			newRow = []
			for impIndex, importance in enumerate(importances):
				if ( importance > threshold ) :
					newRow.append(row[impIndex])
			testNew.append(newRow)	
				
	csv_io.write_delimited_file("PreProcessData/training_PreProcess4.csv", trainNew)		
	csv_io.write_delimited_file("PreProcessData/test_PreProcess4.csv", testNew)
	
			
								
if __name__=="__main__":
	PreProcess4()
#!/usr/bin/env python


import string

from sklearn import svm
import csv_io
import math
from math import log
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import scipy
import numpy as np
import datetime

def toFloat(str):
	return float(str)


def PreProcess3():

	trainBase = csv_io.read_data("PreProcessData/training_PreProcess2_temp.csv", False)
	test = csv_io.read_data("PreProcessData/test_PreProcess2_temp.csv", False)


	
	target = [x[0] for x in trainBase]
	train = [x[1:] for x in trainBase]

	NumFeatures = 200
	
	#clf = RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='gini',compute_importances=True)
	chi = chi2(train, target)
	print "Training"
	#clf.fit(train, target)
		
	
	
	chi = SelectKBest(chi2, k=NumFeatures).fit(train, target)
	print chi.get_support(indices=True)
	print chi.transform(X), np.array(train)[:, [0]]
	
	
	
	return
		
		
	trainNew = []
	testNew = []

	
	
	
	
		
	print "Computing Importances"
	importances = clf.feature_importances_
	#print importances
	importancesTemp = sorted(importances, reverse=True)
	print len(importancesTemp), "importances"
				
	if ( len(importancesTemp) > NumFeatures):
		threshold = importancesTemp[NumFeatures]
		#print "Sorted and deleted importances"
		#print importancesTemp

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
					#print impIndex, len(importances)
					newRow.append(row[impIndex])
			testNew.append(newRow)	
				
	csv_io.write_delimited_file("PreProcessData/training_PreProcess2_chi.csv", trainNew)		
	csv_io.write_delimited_file("PreProcessData/test_PreProcess2_chi.csv", testNew)
	
			
								
if __name__=="__main__":
	PreProcess3()
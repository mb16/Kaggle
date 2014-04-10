#!/usr/bin/env python


import string

from sklearn import svm
import csv_io
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

import gc

import shutil

def toFloat(str):
	return float(str)


def PreProcess5(N_Features):

	trainBase = csv_io.read_data("PreProcessData/training_PreProcess3.csv", skipFirstLine = False, split = "\t")
	
	shutil.copy2("PreProcessData/DataClassList3.csv", "PreProcessData/DataClassList5Base.csv")	
	
	target = [x[0] for x in trainBase]
	train = [x[1:] for x in trainBase]
	
	DataClassList = csv_io.read_data("PreProcessData/DataClassList5Base.csv", False)
	
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
	scaler = preprocessing.Scaler().fit(trainPre)
	trainScaled = scaler.transform(trainPre)
	#testScaled = scaler.transform(testPre)	
	
	

	#clf = RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='gini',compute_importances=True)
	clf = RandomForestRegressor(n_estimators=15, n_jobs=1,compute_importances=True) 
	#clf = ExtraTreesClassifier(n_estimators=150, min_density=0.02, n_jobs=1, criterion='gini', bootstrap=True,compute_importances=True)
	
	print "Training"
	# producing memory errors, probably too much data.
	# recommend to use linear lasso.
	#est = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
	#selector = RFE(est, 20, step=10)
	#selector = selector.fit(trainScaled, target)
	#print selector.support_
	#print selector.ranking_
	#return
	
	#trainPost = selector.transform(trainPre)
	#testPost = selector.transform(testPre)
	
	clf.fit(trainScaled, targetPre)
		
	trainNew = []
	testNew = []

		
	print "Computing Importances"
	importances = clf.feature_importances_

	
	

	DataClassListNew = []
	for DataIndex, DataClass in enumerate(DataClassList):
		print DataClass[0], importances[DataIndex];
		DataClassListNew.append([DataClass[0], importances[DataIndex]])
		
	csv_io.write_delimited_file("PreProcessData/DataClassList_Importances_Base.csv", DataClassListNew)
	
	DataClassListNew_temp = sorted(DataClassListNew, key=operator.itemgetter(1), reverse=True)  
	csv_io.write_delimited_file("PreProcessData/DataClassList_Importances_sorted_Base.csv", DataClassListNew_temp)

	
	
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
			
			
		print "Saving Train"	
		csv_io.write_delimited_file("PreProcessData/training_PreProcess5_Base_" + str(NumFeatures) + ".csv", trainNew, delimiter="\t")	
		# -------
		
		train = []
		trainNew = []
		trainBase = []
		
		gc.collect()
			
		print "Load Test"
		test = csv_io.read_data("PreProcessData/test_PreProcess4.csv", skipFirstLine = False, split = "\t")	
	
		for row in test:
			newRow = []
			for impIndex, importance in enumerate(importances):
				if ( importance > threshold ) :
					newRow.append(row[impIndex])
			testNew.append(newRow)	
				

	csv_io.write_delimited_file("PreProcessData/test_PreProcess5_Base_" +  str(NumFeatures)  + " .csv", testNew, delimiter="\t")
	
	
								
if __name__=="__main__":
	PreProcess5(40)
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
from sklearn.decomposition import PCA
import operator

import shutil

def toFloat(str):
	return float(str)


def PreProcess4():

	trainBase = csv_io.read_data("PreProcessData/training_PreProcess3.csv", skipFirstLine = False, split = "\t")
	test = csv_io.read_data("PreProcessData/test_PreProcess3.csv", skipFirstLine = False, split = "\t")
	shutil.copy2("PreProcessData/DataClassList3.csv", "PreProcessData/DataClassList5_PCA.csv")	
	
	target = [x[0] for x in trainBase]
	train = [x[1:] for x in trainBase]
	
	DataClassList = csv_io.read_data("PreProcessData/DataClassList5_PCA.csv", False)
	
	print "Data len: ", len(train[0])
	print "DataClassList len: ", len(DataClassList)

	NumFeatures = 40

	
	print "Scaling"
	targetPre = [x[0] for x in trainBase]
	trainPre = [x[1:] for x in trainBase]
	testPre = [x[0:] for x in test]
	#print trainPre[0]
	scaler = preprocessing.Scaler().fit(trainPre)
	trainScaled = scaler.transform(trainPre)
	testScaled = scaler.transform(testPre)	
	
	

	#clf = RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='gini',compute_importances=True)
	#clf = RandomForestRegressor(n_estimators=25, n_jobs=1,compute_importances=True) 
	#clf = ExtraTreesClassifier(n_estimators=150, min_density=0.02, n_jobs=1, criterion='gini', bootstrap=True,compute_importances=True)
	
	clf = PCA(n_components=NumFeatures)
	
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
	
	clf.fit(trainScaled, target)
		
	trainNew = []
	testNew = []

		
	print "Computing Importances"
	importances = clf.explained_variance_ratio_

	
	

	#DataClassListNew = []
	#for DataIndex, DataClass in enumerate(DataClassList):
	#	print DataClass[0], importances[DataIndex];
	#	DataClassListNew.append([DataClass[0], importances[DataIndex]])
		
	csv_io.write_delimited_file("PreProcessData/DataClassList_Importances_PCA.csv", importances)
	
	DataClassListNew_temp = sorted(importances, key=operator.itemgetter(1), reverse=True)  
	csv_io.write_delimited_file("PreProcessData/DataClassList_Importances_sorted_PCA.csv", DataClassListNew_temp)

	
	
	importancesTemp = sorted(importances, reverse=True)
	print len(importancesTemp), "importances"
	
	
	trainNew = clf.transform(trainScaled)
	testNew = clf.transform(testScaled)

	
	#if ( len(importancesTemp) > NumFeatures):
	#	threshold = importancesTemp[NumFeatures]

	#	print "Importance threshold: ", threshold

	#	rowIndex = 0
	#	for row in train:
	#		newRow = []
	#		for impIndex, importance in enumerate(importances):
	#			if ( impIndex == 0):
	#				newRow.append(target[rowIndex])
	#			if ( importance > threshold ):	
	#				newRow.append(row[impIndex])
	#		trainNew.append(newRow)	
	#		rowIndex += 1
			
			
	#	for row in test:
	#		newRow = []
	#		for impIndex, importance in enumerate(importances):
	#			if ( importance > threshold ) :
	#				newRow.append(row[impIndex])
	#		testNew.append(newRow)	
				
	csv_io.write_delimited_file("PreProcessData/training_PreProcess5_PCA.csv", trainNew, delimiter="\t")		
	csv_io.write_delimited_file("PreProcessData/test_PreProcess5_PCA.csv", testNew, delimiter="\t")
	
	
								
if __name__=="__main__":
	PreProcess4()
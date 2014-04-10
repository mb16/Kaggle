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
from sklearn.svm import SVC
import scipy
from sklearn import preprocessing
import datetime
from sklearn.linear_model import Lasso, LinearRegression
import operator

import shutil

def toFloat(str):
	return float(str)


def PreProcess4():

	trainBase = csv_io.read_data("PreProcessData/training_PreProcess3.csv", skipFirstLine = False, split = "\t")
	test = csv_io.read_data("PreProcessData/test_PreProcess3.csv", skipFirstLine = False, split = "\t")
	shutil.copy2("PreProcessData/DataClassList3.csv", "PreProcessData/DataClassList4.csv")	
	
	target = [x[0] for x in trainBase]
	train = [x[1:] for x in trainBase]
	
	DataClassList = csv_io.read_data("PreProcessData/DataClassList4.csv", False)
	
	print "Data len: ", len(train[0])
	print "DataClassList len: ", len(DataClassList)
	#return
	
	# this seems about optimal, but has not been tuned on latest improvements.
	NumFeatures = 40
	# NOTE going from 30 to 20 features on KNN5 set has almost no effect.  Down to 15 is significant loss.
	# for GBM at 6 and 400 30 is 3.01 and 30 3.05.
	
	print "Scaling"
	targetPre = [x[0] for x in trainBase]
	trainPre = [x[1:] for x in trainBase]
	testPre = [x[0:] for x in test]
	#print trainPre[0]
	scaler = preprocessing.Scaler().fit(trainPre)
	trainScaled = scaler.transform(trainPre)
	#testScaled = scaler.transform(testPre)	
	
	

	#clf = RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='gini',compute_importances=True)
	#clf = RandomForestRegressor(n_estimators=25, n_jobs=1,compute_importances=True) 
	#clf = ExtraTreesClassifier(n_estimators=150, min_density=0.02, n_jobs=1, criterion='gini', bootstrap=True,compute_importances=True)
	
	print "Training"
	# producing memory errors, probably too much data.
	# recommend to use linear lasso.
	#est = Lasso(fit_intercept=True, normalize=False, copy_X=True)
	est = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
	#est = SVC(kernel='linear')
	selector = RFE(est, 1, step=1)
	selector = selector.fit(trainScaled[1:10000], target[1:10000])
	#print selector.support_
	print selector.ranking_
	#return
	
	#trainPost = selector.transform(trainPre)
	#testPost = selector.transform(testPre)
	
	#clf.fit(trainScaled, target)
		
	trainNew = []
	testNew = []

		
	#print "Computing Importances"
	#importances = clf.feature_importances_

	
	

	DataClassListNew = []
	for DataIndex, DataClass in enumerate(DataClassList):
		print DataClass[0], selector.ranking_[DataIndex];
		DataClassListNew.append([DataClass[0], selector.ranking_[DataIndex]])
		
	csv_io.write_delimited_file("PreProcessData/DataClassList_Importances_RFE_" + str(NumFeatures) + ".csv", DataClassListNew)
	
	DataClassListNew_temp = sorted(DataClassListNew, key=operator.itemgetter(1))  # , reverse=True
	csv_io.write_delimited_file("PreProcessData/DataClassList_Importances_RFE_sorted_" + str(NumFeatures) + ".csv", DataClassListNew_temp)

	
	
	#importancesTemp = sorted(importances, reverse=True)
	#print len(importancesTemp), "importances"
				
	if ( len(selector.ranking_) > NumFeatures):
		#threshold = importancesTemp[NumFeatures]
		
		threshold = NumFeatures
		print "Importance threshold: ", threshold

		rowIndex = 0
		for row in train:
			newRow = []
			for impIndex, importance in enumerate(selector.ranking_):
				if ( impIndex == 0):
					newRow.append(target[rowIndex])
				if ( importance < threshold ):	
					newRow.append(row[impIndex])
			trainNew.append(newRow)	
			rowIndex += 1
			
			
		for row in test:
			newRow = []
			for impIndex, importance in enumerate(selector.ranking_):
				if ( importance < threshold ) :
					newRow.append(row[impIndex])
			testNew.append(newRow)	
				
	csv_io.write_delimited_file("PreProcessData/training_PreProcess4_RFE_" + str(NumFeatures) + ".csv", trainNew, delimiter="\t")		
	csv_io.write_delimited_file("PreProcessData/test_PreProcess4_RFE_" + str(NumFeatures) + ".csv", testNew, delimiter="\t")
	
	
								
if __name__=="__main__":
	PreProcess4()
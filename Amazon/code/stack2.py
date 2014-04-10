#!/usr/bin/env python

import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp
import cPickle as pickle

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, SGDClassifier, LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier
from sklearn import cross_validation
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
import csv_io

model = "Full"

trainSet = pd.read_csv('../train.csv')
testSet = pd.read_csv('../test.csv')

trainTarget = trainSet[['ACTION']]
trainData = trainSet[['RESOURCE','MGR_ID','ROLE_ROLLUP_1','ROLE_ROLLUP_2','ROLE_DEPTNAME','ROLE_TITLE','ROLE_FAMILY_DESC','ROLE_FAMILY']]

testData = testSet[['RESOURCE','MGR_ID','ROLE_ROLLUP_1','ROLE_ROLLUP_2','ROLE_DEPTNAME','ROLE_TITLE','ROLE_FAMILY_DESC','ROLE_FAMILY']]


print trainTarget
print trainData

dataCols = ['RESOURCE','MGR_ID','ROLE_ROLLUP_1','ROLE_ROLLUP_2','ROLE_DEPTNAME','ROLE_TITLE','ROLE_FAMILY_DESC','ROLE_FAMILY']

data = np.unique(trainData[dataCols].values) #[nan, A, B, C, D, F]

lookup = {}
for i, val in enumerate(data):
	lookup[val] = i

	
finalTrain = np.zeros((len(trainTarget),len(data)), dtype=np.int)
finalTest = np.zeros((len(testSet[['RESOURCE']]),len(data)), dtype=np.int)	


for c in dataCols:
	print c
	rowCounter = 0
	for val in trainData[c]:
	
		if lookup.has_key(val):
			#print rowCounter, val, lookup[val]
			finalTrain[rowCounter, lookup[val]] = 1
		rowCounter += 1

for c in dataCols:
	print c
	rowCounter = 0
	for val in testData[c]:
	
		if lookup.has_key(val):
			#print rowCounter, val, lookup[val]
			finalTest[rowCounter, lookup[val]] = 1
		rowCounter += 1
		

finalTrainSparse = sp.csr_matrix(finalTrain)
finalTestSparse = sp.csr_matrix(finalTest)

#with open('train.dat', 'wb') as outfile:
#    pickle.dump(finalTrainSparse, outfile, pickle.HIGHEST_PROTOCOL)
#with open('test.dat', 'wb') as outfile:
#    pickle.dump(finalTestSparse, outfile, pickle.HIGHEST_PROTOCOL)
	
	
#with open('train.dat', 'rb') as infile:
#    finalTrainSparse = pickle.load(infile)
#with open('test.dat', 'rb') as infile:
#    finalTestSparse = pickle.load(infile)	
	
#print finalTrainSparse
	

#km = KMeans(n_clusters=10, init='k-means++', n_init=100, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=1).fit(trainPre)
		
	
clfs = [
LogisticRegression(penalty='l2', dual=False, tol=0.0000001, C=1.0, fit_intercept=False, intercept_scaling=1, class_weight=None, random_state=None)
]
# 0.85 SVC(C=10.0, kernel='rbf', degree=3, gamma=0.001, coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1),
# 0.86?? LogisticRegression(penalty='l2', dual=False, tol=0.0000001, C=1.0, fit_intercept=False, intercept_scaling=1, class_weight=None, random_state=None)]


avg = 0
NumFolds = 5
predicted_list = []


print "Data size: ", len(trainTarget), len(testSet[['RESOURCE']])
dataset_blend_train = np.zeros((len(trainTarget), len(clfs)))
dataset_blend_test = np.zeros((len(testSet[['RESOURCE']]), len(clfs)))


print "Begin Training"
		
for ExecutionIndex, clf in enumerate(clfs):
	print clf
	avg = 0
	
	predicted_list = []			
	dataset_blend_test_set = np.zeros((len(testSet[['RESOURCE']]), NumFolds))
		
	foldCount = 0

	Folds = cross_validation.KFold(len(trainTarget), n_folds=NumFolds, indices=True)
	for train_index, test_index in Folds:

		#target = [targetPre[i] for i in train_index]
		#train = [trainPre[i] for i in train_index]
		target = np.asarray(trainTarget)[train_index, : ]
		train = finalTrainSparse[train_index, : ]	
		train = train.todense()
			
		#train = trainPre.tocsr()[train_index,:]
			
		#targetTest = [targetPre[i] for i in test_index]	
		#trainTest = [trainPre[i] for i in test_index]	
		targetTest = np.asarray(trainTarget)[test_index, : ]
		trainTest = finalTrainSparse[test_index, : ]			
		trainTest = trainTest.todense()
		
		#trainTest = trainPre.tocsr()[test_index,:]

		print
		print "Iteration: ", foldCount
		#print "LEN: ", len(train), len(target)
		
		#train = km.transform(train)
		#trainTest = km.transform(trainTest)
		
		clf.fit(train, target)
		print "Predict"
		#prob = clf.predict(trainTest) 
		prob = clf.predict_proba(trainTest) 
		#print "Score", prob, prob[:,1]
		dataset_blend_train[test_index, ExecutionIndex] = prob[:,1]

	
			
		fpr, tpr, thresholds = metrics.roc_curve(targetTest, prob[:,1], pos_label=1)
		auc = metrics.auc(fpr,tpr)
		print "Score: ", auc
			

		avg += 	auc/NumFolds

		predicted_probs = clf.predict_proba(finalTestSparse) 	
		#predicted_list.append([x[1] for x in predicted_probs])	
		dataset_blend_test_set[:, foldCount] = predicted_probs[:,1]
		
				
		foldCount = foldCount + 1
		
		#break	
		
	dataset_blend_test[:,ExecutionIndex] = dataset_blend_test_set.mean(1)  
		

		
	now = datetime.datetime.now()

	csv_io.write_delimited_file_single_plus_index("../predictions/Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_test_set.mean(1))
	csv_io.write_delimited_file_single("../predictions/Target_Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_train[:,ExecutionIndex] )		
	csv_io.write_delimited_file("../predictions/RunLog.csv", [now.strftime("%Y %m %d %H %M %S"), str(avg), str(clf), str(NumFolds), model, "", ""], filemode="a",delimiter=",")
		
		
	print "------------------------Average: ", avg





#!/usr/bin/env python

from sklearn import svm
import csv_io
import math
from math import log
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import scipy

import datetime
import random


	
def run_stack(SEED):

	print "Running GB, RF, ET stack."

	trainBase = csv_io.read_data("PreProcessData/training_PreProcess4.csv", False)
	test = csv_io.read_data("PreProcessData/test_PreProcess4.csv", False)
	
	random.seed(SEED)
	random.shuffle(trainBase)
	
	avg = 0
	NumFolds = 10 # 5 is good, but 10 yeilds a better mean since outliers are less significant. (note, predictions are less reliable when using 10).

	NumFeatures = 1000

	predicted_list = []
	
	spanDistance = 12
	bootstrapLists = []

	# use this for quick runs.
	clfs = [GradientBoostingClassifier(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
			GradientBoostingClassifier(learn_rate=0.02, subsample=0.2, max_depth=8, n_estimators=125),
			RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='gini'),
			RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='entropy'),
			ExtraTreesClassifier(n_estimators=150, min_density=0.2, n_jobs=1, criterion='gini', bootstrap=True, random_state=1),
			ExtraTreesClassifier(n_estimators=150, min_density=0.2, n_jobs=1, criterion='entropy', bootstrap=True, random_state=5)]	
	
		# use this for quick runs.
	# clfs = [GradientBoostingClassifier(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50, random_state=166),
			# GradientBoostingClassifier(learn_rate=0.02, subsample=0.2, max_depth=8, n_estimators=125, random_state=551),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=80, random_state=441),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=80, random_state=331),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=4, n_estimators=80, random_state=221),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=120, random_state=91),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=120, random_state=81),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=4, n_estimators=120, random_state=71),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=160, random_state=61),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=160, random_state=51),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=4, n_estimators=160, random_state=41),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=1, n_estimators=200, random_state=31),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=200, random_state=21),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=200, random_state=10),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=4, n_estimators=200, random_state=19),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=1, n_estimators=240, random_state=18),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=240, random_state=17),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=240, random_state=16),			
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=1, n_estimators=280, random_state=15),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=280, random_state=14),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=280, random_state=13),			
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=1, n_estimators=320, random_state=12),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=320, random_state=11),
			# RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='gini'),
			# RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='entropy'),
			# ExtraTreesClassifier(n_estimators=150, min_density=0.2, n_jobs=1, criterion='gini', bootstrap=True, random_state=1),
			# ExtraTreesClassifier(n_estimators=150, min_density=0.2, n_jobs=1, criterion='entropy', bootstrap=True, random_state=5)]	
	
	
	
	# use this for quick runs.
	# clfs = [GradientBoostingClassifier(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
			# GradientBoostingClassifier(learn_rate=0.02, subsample=0.2, max_depth=8, n_estimators=125),
			# RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='gini'),
			# RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='entropy')
			# ]	
			

	# clfs = [ExtraTreesClassifier(n_estimators=150, min_density=0.2, n_jobs=1, criterion='gini', bootstrap=True, random_state=1),
			# ExtraTreesClassifier(n_estimators=150, min_density=0.09, n_jobs=1, criterion='gini', bootstrap=True, random_state=2),
			# ExtraTreesClassifier(n_estimators=150, min_density=0.2, n_jobs=1, criterion='entropy', bootstrap=True, random_state=5),
			# ExtraTreesClassifier(n_estimators=150, min_density=0.09, n_jobs=1, criterion='entropy', bootstrap=True, random_state=6),
			# GradientBoostingClassifier(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
			# GradientBoostingClassifier(learn_rate=0.02, subsample=0.2, max_depth=8, n_estimators=125),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=80),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=80),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=4, n_estimators=80),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=120),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=120),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=4, n_estimators=120),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=160),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=160),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=4, n_estimators=160),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=1, n_estimators=200),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=200),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=200),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=4, n_estimators=200),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=1, n_estimators=240),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=240),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=240),			
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=1, n_estimators=280),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=280),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=280),			
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=1, n_estimators=320),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=320),
			# RandomForestClassifier(n_estimators=150, min_density=0.2, n_jobs=1, criterion='gini', bootstrap=True, random_state=1),
			# RandomForestClassifier(n_estimators=150, min_density=0.09, n_jobs=1, criterion='gini', bootstrap=True, random_state=2),
			# RandomForestClassifier(n_estimators=150, min_density=0.2, n_jobs=1, criterion='entropy', bootstrap=True, random_state=5),
			# RandomForestClassifier(n_estimators=150, min_density=0.05, n_jobs=1, criterion='entropy', bootstrap=True, random_state=7)]
			
			
	# full algorithm stack.
	# clfs = [ExtraTreesClassifier(n_estimators=150, min_density=0.2, n_jobs=1, criterion='gini', bootstrap=True, random_state=1),
			# ExtraTreesClassifier(n_estimators=150, min_density=0.09, n_jobs=1, criterion='gini', bootstrap=True, random_state=2),
			# ExtraTreesClassifier(n_estimators=150, min_density=0.05, n_jobs=1, criterion='gini', bootstrap=True, random_state=3),
			# ExtraTreesClassifier(n_estimators=150, min_density=0.02, n_jobs=1, criterion='gini', bootstrap=True, random_state=4),
			# ExtraTreesClassifier(n_estimators=150, min_density=0.2, n_jobs=1, criterion='entropy', bootstrap=True, random_state=5),
			# ExtraTreesClassifier(n_estimators=150, min_density=0.09, n_jobs=1, criterion='entropy', bootstrap=True, random_state=6),
			# ExtraTreesClassifier(n_estimators=150, min_density=0.05, n_jobs=1, criterion='entropy', bootstrap=True, random_state=7),
			# ExtraTreesClassifier(n_estimators=150, min_density=0.02, n_jobs=1, criterion='entropy', bootstrap=True, random_state=8),
			# GradientBoostingClassifier(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
			# GradientBoostingClassifier(learn_rate=0.02, subsample=0.2, max_depth=8, n_estimators=125),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=80),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=80),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=4, n_estimators=80),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=120),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=120),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=4, n_estimators=120),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=160),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=160),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=4, n_estimators=160),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=1, n_estimators=200),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=200),
			# GradientBoostingClassifier(lesarn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=200),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=4, n_estimators=200),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=1, n_estimators=240),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=240),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=240),			
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=1, n_estimators=280),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=280),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=3, n_estimators=280),			
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=1, n_estimators=320),
			# GradientBoostingClassifier(learn_rate=0.01, subsample=0.2, max_depth=2, n_estimators=320),
			# RandomForestClassifier(n_estimators=150, min_density=0.2, n_jobs=1, criterion='gini', bootstrap=True, random_state=1),
			# RandomForestClassifier(n_estimators=150, min_density=0.09, n_jobs=1, criterion='gini', bootstrap=True, random_state=2),
			# RandomForestClassifier(n_estimators=150, min_density=0.05, n_jobs=1, criterion='gini', bootstrap=True, random_state=3),
			# RandomForestClassifier(n_estimators=150, min_density=0.02, n_jobs=1, criterion='gini', bootstrap=True, random_state=4),
			# RandomForestClassifier(n_estimators=150, min_density=0.2, n_jobs=1, criterion='entropy', bootstrap=True, random_state=5),
			# RandomForestClassifier(n_estimators=150, min_density=0.09, n_jobs=1, criterion='entropy', bootstrap=True, random_state=6),
			# RandomForestClassifier(n_estimators=150, min_density=0.05, n_jobs=1, criterion='entropy', bootstrap=True, random_state=7),
			# RandomForestClassifier(n_estimators=150, min_density=0.02, n_jobs=1, criterion='entropy', bootstrap=True, random_state=8)]
	

	
	print "Data size: ", len(trainBase), len(test)
	dataset_blend_train = np.zeros((len(trainBase), len(clfs)))
	dataset_blend_test = np.zeros((len(test), len(clfs)))
	

	trainNew = []
	trainTestNew = []
	testNew = []
	trainNewSelect = []
	trainTestNewSelect = []
	testNewSelect = []
	


	for ExecutionIndex, clf in enumerate(clfs):
		print clf
		avg = 0
	
		predicted_list = []
			
		dataset_blend_test_set = np.zeros((len(test), NumFolds))
		
		foldCount = 0

		Folds = cross_validation.StratifiedKFold([trainBase[i][0] for i in range(len(trainBase))], k=NumFolds, indices=True)
		for train_index, test_index in Folds:

			trainBaseTemp = [trainBase[i] for i in train_index]
			target = [x[0] for x in trainBaseTemp]
			train = [x[1:] for x in trainBaseTemp]
	
			testBaseTemp = [trainBase[i] for i in test_index]
			targetTest = [x[0] for x in testBaseTemp]
			trainTest = [x[1:] for x in testBaseTemp]
	
	
			test = [x[0:] for x in test]
	
			print
			print "Iteration: ", foldCount
			print "LEN: ", len(train), len(target)
			
			clf.fit(train, target)
			prob = clf.predict_proba(trainTest) 
			
			dataset_blend_train[test_index, ExecutionIndex] = prob[:,1] 
			
	
			probSum = 0
			totalOffByHalf = 0
			totalPositive = 0
			totalPositiveOffByHalf = 0
			totalPositivePredictions = 0
			
			for i in range(0, len(prob)):
				probX = prob[i][1] 
				if ( probX > 0.999):
					probX = 0.999;		
				if ( probX < 0.001):
					probX = 0.001;

				probSum += int(targetTest[i])*log(probX)+(1-int(targetTest[i]))*log(1-probX)
				if ( math.fabs(probX - int(targetTest[i])) > 0.5 ):
					totalOffByHalf = totalOffByHalf + 1		
			
				if ( int(targetTest[i]) == 1 ):
					totalPositive = totalPositive + 1
				if ( int(targetTest[i]) == 1 and probX < 0.5):
					totalPositiveOffByHalf = totalPositiveOffByHalf + 1
				if (probX > 0.5):
					totalPositivePredictions = totalPositivePredictions + 1			
			
			print
			print "Stats:"
			print "Total Off By > 0.5 ", totalOffByHalf
			print "Total Positive ", totalPositive
			print "Total Positive Off By Half ", totalPositiveOffByHalf
			print "Total Positive Predictions ", totalPositivePredictions
			print -probSum/len(prob)
	
 
			avg += 	(-probSum/len(prob))/NumFolds

			predicted_probs = clf.predict_proba(test) 					

			predicted_list.append([x[1] for x in predicted_probs])
		
			dataset_blend_test_set[:, foldCount] = predicted_probs[:,1]
		
			foldCount = foldCount + 1
		
		dataset_blend_test[:,ExecutionIndex] = dataset_blend_test_set.mean(1)  
		
		
		print "------------------------Average: ", avg

		

	return dataset_blend_train, dataset_blend_test
							
	
	
if __name__=="__main__":
	run_stack(448)
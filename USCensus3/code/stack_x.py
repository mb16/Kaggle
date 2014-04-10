#!/usr/bin/env python

from sklearn import svm
import csv_io
import math
from math import log
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import scipy

import datetime
import random

from sklearn import preprocessing
	
def run_stack(SEED):

	model = "Long-Lat KNN5 - 50 Features"

	print "Running GB, RF, ET stack."

	trainBase = csv_io.read_data("PreProcessData/training_PreProcess4_50.csv", skipFirstLine = False, split = "\t")
	test = csv_io.read_data("PreProcessData/test_PreProcess4_50.csv", skipFirstLine = False, split = "\t")
	weights = csv_io.read_data("PreProcessData/Weights.csv", skipFirstLine = False)

	
	#random.seed(SEED)
	#random.shuffle(trainBase)
	
	avg = 0
	NumFolds = 5 # 5 is good, but 10 yeilds a better mean since outliers are less significant. (note, predictions are less reliable when using 10).


	predicted_list = []
	bootstrapLists = []

	# use this for quick runs.
	# note RF with 150 crashes on 30 features
	# GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
	# GradientBoostingRegressor(learn_rate=0.02, subsample=0.2, max_depth=8, n_estimators=125),
	# RandomForestRegressor(n_estimators=100, n_jobs=1),
	#RandomForestRegressor(n_estimators=75, n_jobs=1),
	# clfs = [ExtraTreesRegressor(n_estimators=50, min_density=0.2, n_jobs=1, bootstrap=True, random_state=1),
		# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=2**-5.5, kernel='rbf', probability=False, shrinking=True, tol=0.001, verbose=False)
		# ]	
	#knn 5 at 3.45
	#knn 15 at 3.31
	#knn 25 at 3.30
	#knn 40 at 3.31
	# KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=False, p=2),
	# KNeighborsRegressor(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=False, p=2),
	# KNeighborsRegressor(n_neighbors=25, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=False, p=2),
	# LinearRegression at 3.77
	# Ridge at 3.77
	# SGD 4.23
	#Gauss at 13
	# LinearRegression(fit_intercept=True, normalize=False, copy_X=True),
	# Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, tol=0.001),
	# SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, rho=0.84999999999999998, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.10000000000000001, p=None, seed=0, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False),
	# GaussianNB()
	# clfs = [KNeighborsRegressor(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=False, p=2),
		 # KNeighborsRegressor(n_neighbors=25, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=False, p=2),KNeighborsRegressor(n_neighbors=35, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=False, p=2)	
		# ]
		
	# GB, 125 est is minimum, score is bad below this, explore higher and other dimensions. ******************
	# clfs = [GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=100, random_state=166),
			# GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=8, n_estimators=100, random_state=166),
			# GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=10, n_estimators=100, random_state=166),
			# GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=200, random_state=166),
			# GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=8, n_estimators=200, random_state=166),GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=10, n_estimators=200, random_state=166)
			# ]	
			
	# about 1 hour run time, and 3.10 score.		
	#GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=400, random_state=166)
	# about 2 hours run time at 3.05
	#GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=8, n_estimators=400, random_state=166)
	# about 2 hours run time at 3.06
	#GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=800, random_state=166)
	# about 4 hours run time at 3.06
	#GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=8, n_estimators=800, random_state=166)	
	
	clfs = [GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=3000, random_state=166)
			]		
	
	
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
	
	
	
	# use this for quick runs.  reduced estimators to 50
	# clfs = [SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
        # gamma=2**-5.5, kernel='rbf', probability=False, shrinking=True,
        # tol=0.001, verbose=False)
			# ]	
			
	#GradientBoostingRegressor(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
	#ExtraTreesRegressor(n_estimators=50, min_density=0.2, n_jobs=1, bootstrap=True, random_state=1)
	
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
	
	print "Scaling"
	targetPre = [x[0] for x in trainBase]
	trainPre = [x[1:] for x in trainBase]
	testPre = [x[0:] for x in test]
	#print trainPre[0]
	scaler = preprocessing.Scaler().fit(trainPre)
	trainScaled = scaler.transform(trainPre)
	testScaled = scaler.transform(testPre)	

	#print scaler.mean_
	#print scaler.std_
	print "Begin Training"
	
	
	for ExecutionIndex, clf in enumerate(clfs):
		print str(clf)
		avg = 0
	
		predicted_list = []
			
		dataset_blend_test_set = np.zeros((len(test), NumFolds))

		
		foldCount = 0

		
		#Stratified for classification...[trainBase[i][0] for i in range(len(trainBase))]
		Folds = cross_validation.KFold(len(trainBase), k=NumFolds, indices=True)
		for train_index, test_index in Folds:

			#trainBaseTemp = [trainBase[i] for i in train_index]
			#target = [x[0] for x in trainBaseTemp]
			#train = [x[1:] for x in trainBaseTemp]
	
			#testBaseTemp = [trainBase[i] for i in test_index]
			#targetTest = [x[0] for x in testBaseTemp]
			#trainTest = [x[1:] for x in testBaseTemp]
		
			#test = [x[0:] for x in test]
	
			target = [targetPre[i] for i in train_index]
			train = [trainScaled[i] for i in train_index]
			
			targetTest = [targetPre[i] for i in test_index]	
			trainTest = [trainScaled[i] for i in test_index]	
	
			print
			print "Iteration: ", foldCount
			print "LEN: ", len(train), len(target)
			
			clf.fit(train, target)
			prob = clf.predict(trainTest) 
			
			dataset_blend_train[test_index, ExecutionIndex] = prob



	
			probSum = 0
			weightSum = 0
			# totalOffByHalf = 0
			# totalPositive = 0
			# totalPositiveOffByHalf = 0
			# totalPositivePredictions = 0
			
			for i in range(0, len(prob)):
				probX = prob[i]

				probSum += weights[test_index[i]][0] * math.fabs(targetTest[i] - probX)
				weightSum += weights[test_index[i]][0] 
				#print "Weight", weights[test_index[i]][0], "Index: ",i, "Test_Index: ",test_index[i] , "Actual: ", targetTest[i], "Predicted: ", probX
				
				# log loss cal
				#probSum += int(targetTest[i])*log(probX)+(1-int(targetTest[i]))*log(1-probX)
				# if ( math.fabs(probX - int(targetTest[i])) > 0.5 ):
					# totalOffByHalf = totalOffByHalf + 1		
			
				# if ( int(targetTest[i]) == 1 ):
					# totalPositive = totalPositive + 1
				# if ( int(targetTest[i]) == 1 and probX < 0.5):
					# totalPositiveOffByHalf = totalPositiveOffByHalf + 1
				# if (probX > 0.5):
					# totalPositivePredictions = totalPositivePredictions + 1			
			
			# print
			# print "Stats:"
			# print "Total Off By > 0.5 ", totalOffByHalf
			# print "Total Positive ", totalPositive
			# print "Total Positive Off By Half ", totalPositiveOffByHalf
			# print "Total Positive Predictions ", totalPositivePredictions
			#print -probSum/len(prob)
			print "Score: ", probSum/weightSum
 
			avg += 	(probSum/weightSum)/NumFolds

			predicted_probs = clf.predict(testScaled) 	
			#predicted_list.append([x[1] for x in predicted_probs])	
			dataset_blend_test_set[:, foldCount] = predicted_probs #[0]
		
				
			foldCount = foldCount + 1
		
		dataset_blend_test[:,ExecutionIndex] = dataset_blend_test_set.mean(1)  
		
		#print "Saving NP"
		#np.savetxt('temp/dataset_blend_test_set.txt', dataset_blend_test_set)
		#np.savetxt('temp/dataset_blend_test_set.mean.txt', dataset_blend_test_set.mean(1) )
		#np.savetxt('temp/dataset_blend_test.txt', dataset_blend_test)
		#print "Done Saving NP"
		
		now = datetime.datetime.now()
		#print dataset_blend_test_set.mean(1) 
		csv_io.write_delimited_file_single("../predictions_50/Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_test_set.mean(1))
		
		csv_io.write_delimited_file_single("../predictions_50/Target_Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_train[:,ExecutionIndex] )		
		
		csv_io.write_delimited_file("../predictions_40/RunLog.csv", [now.strftime("%Y %m %d %H %M %S"), str(avg), str(clf), str(NumFolds), model, "", ""], filemode="a",delimiter=",")
		
		
		print now
		print "------------------------Average: ", avg

		#np.savetxt('temp/dataset_blend_train.txt', dataset_blend_train)

	return dataset_blend_train, dataset_blend_test
							
	
	
if __name__=="__main__":
	run_stack(448)
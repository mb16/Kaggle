#!/usr/bin/env python

from sklearn import svm
import csv_io
import math
from sklearn import metrics
from math import log
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, SGDClassifier, LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import scipy
from sklearn.preprocessing import OneHotEncoder
import datetime
import random
from sklearn.cluster import KMeans

from sklearn import preprocessing
	
def run_stack(SEED):

	model = ""

	print "Running GB, RF, ET stack."

	trainBase = csv_io.read_data("../train.csv", skipFirstLine = True, split = ",")
	test = csv_io.read_data("../test.csv", skipFirstLine = True, split = ",")


	
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
	#SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, rho=None, seed=None)
	
	# http://stackoverflow.com/questions/15150339/python-memory-error-sklearn-huge-input-data
	#For high dimensional sparse data and many samples, LinearSVC, LogisticRegression, 
	# PassiveAggressiveClassifier or SGDClassifier can be much faster to train for comparable predictive accuracy.
	
	# LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
	# LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None)
	# PassiveAggressiveClassifier(C=1.0, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, loss='hinge', n_jobs=1, random_state=None, warm_start=False)
	# SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, rho=None, seed=None)
	
	clfs = [RandomForestClassifier(n_estimators=500, n_jobs=1, criterion='gini')
	] 	

	# best SVC(C=1000000.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1),
	# best LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1000.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None),

	
	#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,gamma=0.0, kernel='rbf', max_iter=-1, probability=False, shrinking=True,tol=0.001, verbose=False)
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
	#targetPre = [x[0] for x in trainBase]
	#trainPre = [x[1:] for x in trainBase]
	#trainPreTemp = [x[1:] for x in trainBase]
	#testPre = [x[1:] for x in test]

	targetPre = [int(x[0]) for x in trainBase]
	trainPre = [[int(i) for i in x[1:]] for x in trainBase]
	trainPreTemp = [[int(i) for i in x[1:]] for x in trainBase]
	testPre = [[int(i) for i in x[1:]] for x in test]
	
	print "unique: ", len(list(set([x[1] for x in trainBase])))
	
	#enc = OneHotEncoder()
	#print len(trainPreTemp)
	#trainPreTemp.extend(testPre)
	#print len(trainPreTemp)
	#enc.fit(trainPreTemp)
	#print enc.n_values_
	#print enc.feature_indices_
	
	#out = enc.transform(trainPre)
	#trainPre = out#.toarray()
	#print out.shape # len(out), len(out[0])
	
	#out = enc.transform(testPre)
	#testPre = out#.toarray()
	#print out.shape
	
	km = KMeans(n_clusters=10, init='k-means++', n_init=100, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=1).fit(trainPre)
	
	
	
	#return
	
	
	#print trainPre[0]
	#scaler = preprocessing.Scaler().fit(trainPre)
	#trainScaled = scaler.transform(trainPre)
	#testScaled = scaler.transform(testPre)	

	#print scaler.mean_
	#print scaler.std_
	print "Begin Training"
	
	
	for ExecutionIndex, clf in enumerate(clfs):
		print clf
		avg = 0
	
		predicted_list = []
			
		dataset_blend_test_set = np.zeros((len(test), NumFolds))

		
		foldCount = 0

		
		#Stratified for classification...[trainBase[i][0] for i in range(len(trainBase))]
		#Folds = cross_validation.StratifiedKFold(targetPre, n_folds=NumFolds, indices=True)
		Folds = cross_validation.KFold(len(trainBase), n_folds=NumFolds, indices=True)
		for train_index, test_index in Folds:

			#trainBaseTemp = [trainBase[i] for i in train_index]
			#target = [x[0] for x in trainBaseTemp]
			#train = [x[1:] for x in trainBaseTemp]
	
			#testBaseTemp = [trainBase[i] for i in test_index]
			#targetTest = [x[0] for x in testBaseTemp]
			#trainTest = [x[1:] for x in testBaseTemp]
		
			#test = [x[0:] for x in test]
	
			target = [targetPre[i] for i in train_index]
			train = [trainPre[i] for i in train_index]
			
			#train = trainPre.tocsr()[train_index,:]
			
			targetTest = [targetPre[i] for i in test_index]	
			trainTest = [trainPre[i] for i in test_index]	
			
			#trainTest = trainPre.tocsr()[test_index,:]
	
	
			print
			print "Iteration: ", foldCount
			#print "LEN: ", len(train), len(target)
		
			train = km.transform(train)
			trainTest = km.transform(trainTest)
		
			clf.fit(train, target)
			print "Predict"
			prob = clf.predict_proba(trainTest) 
			print "Score"
			dataset_blend_train[test_index, ExecutionIndex] = prob[:,1]



	
			probSum = 0
			weightSum = 0
			# totalOffByHalf = 0
			# totalPositive = 0
			# totalPositiveOffByHalf = 0
			# totalPositivePredictions = 0
			
			fpr, tpr, thresholds = metrics.roc_curve(targetTest, prob[:,1], pos_label=1)
			auc = metrics.auc(fpr,tpr)
			print "Score: ", auc
			
			#for i in range(0, len(prob)):
				#print prob
				#probX = prob[i]

				#probSum += weights[test_index[i]][0] * math.fabs(targetTest[i] - probX)
				#weightSum += weights[test_index[i]][0] 
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
			#print "Score: ", probSum/weightSum
 
			avg += 	auc/NumFolds

			predicted_probs = clf.predict_proba(testPre) 	
			#predicted_list.append([x[1] for x in predicted_probs])	
			dataset_blend_test_set[:, foldCount] = predicted_probs[:,1] #[0]
		
				
			foldCount = foldCount + 1
		
			break
		
		dataset_blend_test[:,ExecutionIndex] = dataset_blend_test_set.mean(1)  
		
		#print "Saving NP"
		#np.savetxt('temp/dataset_blend_test_set.txt', dataset_blend_test_set)
		#np.savetxt('temp/dataset_blend_test_set.mean.txt', dataset_blend_test_set.mean(1) )
		#np.savetxt('temp/dataset_blend_test.txt', dataset_blend_test)
		#print "Done Saving NP"
		
		now = datetime.datetime.now()
		#print dataset_blend_test_set.mean(1) 
		csv_io.write_delimited_file_single_plus_index("../predictions/Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_test_set.mean(1))
		
		csv_io.write_delimited_file_single("../predictions/Target_Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_train[:,ExecutionIndex] )		
		
		csv_io.write_delimited_file("../predictions/RunLog.csv", [now.strftime("%Y %m %d %H %M %S"), str(avg), str(clf), str(NumFolds), model, "", ""], filemode="a",delimiter=",")
		
		
		print "------------------------Average: ", avg

		#np.savetxt('temp/dataset_blend_train.txt', dataset_blend_train)

	return dataset_blend_train, dataset_blend_test
							
	
	
if __name__=="__main__":
	run_stack(448)
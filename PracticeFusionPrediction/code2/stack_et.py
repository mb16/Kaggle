#!/usr/bin/env python

from sklearn import svm
import csv_io
import math
from math import log
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import scipy
import datetime

import random

def SimpleScale(probArray, floor = 0.001, ceiling = 0.999):

	minProb = 0.5
	maxProb = 0.5
	
	# search for min and max probs
	for i in range(0, len(probArray)):
		probX = probArray[i] # [1]
		
		if ( probX  > maxProb ):
			maxProb = probX
		if ( probX  < minProb ):
			minProb = probX
			
	#scale those below 0.5 down to 0 and above 0.5 up to 1		
	for i in range(0, len(probArray)):
		probX = probArray[i] # [1]			
					
		if ( probX < 0.5 ):
			probArray[i] = 0.5 - ((0.5 - probX)/(0.5 - minProb)) * 0.5 
			#print probX, probArray[i]
			
		if ( probX > 0.5 ):
			probArray[i] = 0.5 + ((probX - 0.5)/(maxProb - 0.5)) * 0.5 
			#probArray[i] = ceiling;		
			
		if ( probArray[i] < floor):
			probArray[i] = floor;
		
	print "SimpleScale: ", minProb, maxProb
	
	return probArray		

def mean(numberList):
	if len(numberList) == 0:
		return float('nan')
 
	floatNums = [float(x) for x in numberList]
	return sum(floatNums) / len(numberList)

def getMedian(numericValues):

	theValues = sorted(numericValues)
	if len(theValues) % 2 == 1:
		return theValues[(len(theValues)+1)/2-1]
	else:
		lower = theValues[len(theValues)/2-1]
		upper = theValues[len(theValues)/2]
		
	return (float(lower + upper)) / 2  
	
	
def meanSpan(numberList, N):

	a = sorted(numberList)
	#print a
	span = []
	minSpan = 100.0
	minSpanIndex = 0
	spanSum = 0.0
	
	#print a

	for i in range(0,len(a)-N + 1):
		span.append(a[i + N - 1] - a[i])
		
	for i in range(0,len(span)):
		if ( span[i] < minSpan):
			minSpan = span[i]
			minSpanIndex = i
		
	print a[minSpanIndex:minSpanIndex+N]
	spanSum = sum(a[minSpanIndex:minSpanIndex+N])	
		
	#print span
	#print spanSum
	#print spanSum/float(N)
	#print minSpan, minSpanIndex	
	
	return spanSum/float(N)	
	
def run_stack(SEED):
	print "Running ET Stack"
	
	trainBase = csv_io.read_data("PreProcessData/training_PreProcess2.csv", False)
	test = csv_io.read_data("PreProcessData/test_PreProcess2.csv", False)
	
	random.seed(SEED)
	random.shuffle(trainBase)
	
	avg = 0
	NumFolds = 10 # should be odd for median

	NumFeatures = 1000

	predicted_list = []
	
	spanDistance = 12
	bootstrapLists = []
	
	#clfs = [GradientBoostingClassifier(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
	#		GradientBoostingClassifier(learn_rate=0.02, subsample=0.2, max_depth=8, n_estimators=125),
	#		RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='gini',compute_importances=True),
    #        RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='entropy',compute_importances=True),
    #        ExtraTreesClassifier(n_estimators=100, n_jobs=1, criterion='gini', compute_importances=True),
    #        ExtraTreesClassifier(n_estimators=100, n_jobs=1, criterion='entropy',compute_importances=True)]
		
	rnd_start = 456
	
	n_estArr = [50] #20,40,80,160,,640,1280,4000,8000,16000
	learn_rArr =  [0.5, 0.25, 0.1, 0.075, 0.05] # DEFAULT 0.10000000000000001
	
	print len(trainBase), len(test)
	dataset_blend_train = np.zeros((len(trainBase), len(n_estArr)*len(learn_rArr)))
	dataset_blend_test = np.zeros((len(test), len(n_estArr)*len(learn_rArr)))
	

	trainNew = []
	trainTestNew = []
	testNew = []
	trainNewSelect = []
	trainTestNewSelect = []
	testNewSelect = []
	
	print "Start Feaure Select"
	#f_classif(np.array([x[1:] for x in trainBase]), np.array([x[0] for x in trainBase]))
	#print "done1"
	#fs = SelectKBest(chi2, k=NumFeatures)
	#fs.fit(scipy.array([x[1:] for x in trainBase]), scipy.array([x[0] for x in trainBase]))
	#fs.fit(np.array([x[1:] for x in trainBase]), np.array([x[0] for x in trainBase]))
	print "End Feaure Select"	
	
	LastClassifier = ""
	ExecutionIndex = 0
	
	#for ExecutionIndex, clf in enumerate(clfs):
	for n_est in n_estArr:
		for learn_r in learn_rArr:
			print "n_est ", n_est, "learn_r " ,learn_r
			clf = ExtraTreesClassifier(n_estimators=n_est, criterion='entropy', max_depth=None, min_samples_split=1, min_samples_leaf=1, min_density=learn_r, max_features='auto', bootstrap=True, compute_importances=False, oob_score=False, n_jobs=1, random_state=None, verbose=0)

			print clf
			avg = 0
		
			predicted_list = []
				
			dataset_blend_test_j = np.zeros((len(test), NumFolds))
			
			foldCount = 0
			
			#print [trainBase[i][0] for i in range(len(trainBase))]
			#Folds = cross_validation.KFold(len(trainBase) - 1, k=NumFolds, indices=True, shuffle=False, random_state=None)		
			Folds = cross_validation.StratifiedKFold([trainBase[i][0] for i in range(len(trainBase))], k=NumFolds, indices=True)
			for train_index, test_index in Folds:

				trainBaseTemp = [trainBase[i] for i in train_index]
				target = [x[0] for x in trainBaseTemp]
				train = [x[1:] for x in trainBaseTemp]
		
				testBaseTemp = [trainBase[i] for i in test_index]
				targetTest = [x[0] for x in testBaseTemp]
				trainTest = [x[1:] for x in testBaseTemp]
		
		

				test = [x[0:] for x in test]
		

				#rf = RandomForestClassifier(n_estimators=n_est, criterion='entropy', max_depth=None, min_samples_split=1, min_samples_leaf=1, min_density=0.10000000000000001, max_features='auto', bootstrap=True, compute_importances=False, oob_score=False, n_jobs=1, random_state=None, verbose=0) # , max_features=None

				print "LEN: ", len(train), len(target)
				

				
				if (False and LastClassifier !=  str(clf)[:10] and (str(clf).startswith( 'RandomForest' ) or str(clf).startswith( 'ExtraTrees' ))) :

					clf.fit(train, target)
				
					LastClassifier = str(clf)[:10]
					print "Computing Importances"
					importances = clf.feature_importances_
					#print importances
					importancesTemp = sorted(importances, reverse=True)
					print len(importancesTemp), "importances"
					
					if ( len(importancesTemp) > NumFeatures):
						threshold = importancesTemp[NumFeatures]
						#print "Sorted and deleted importances"
						#print importancesTemp

						for row in train:
							newRow = []
							for impIndex, importance in enumerate(importances):
								if ( importance > threshold ) :	
									newRow.append(row[impIndex])
							trainNew.append(newRow)	

						for row in trainTest:
							newRow = []
							for impIndex, importance in enumerate(importances):
								if ( importance > threshold ) :
									newRow.append(row[impIndex])
							trainTestNew.append(newRow)	

						for row in test:
							newRow = []
							for impIndex, importance in enumerate(importances):
								if ( importance > threshold ) :
									#print impIndex, len(importances)
									newRow.append(row[impIndex])
							testNew.append(newRow)	
					
					else:
						trainNew = train
						trainTestNew = trainTest
						testNew = test	
				else:
					#trainNew = fs.transform(train)
					#trainTestNew = fs.transform(trainTest)
					#testNew = fs.transform(test)
					trainNew = train
					trainTestNew = trainTest
					testNew = test

				clf.fit(trainNew, target)



				prob = clf.predict_proba(trainTestNew) 
				
				dataset_blend_train[test_index, ExecutionIndex] = prob[:,1] 
				
		
				probSum = 0
				totalOffByHalf = 0
				totalPositive = 0
				totalPositiveOffByHalf = 0
				totalPositivePredictions = 0
				
				for i in range(0, len(prob)):
					probX = prob[i][1] # [1]
					if ( probX > 0.999):
						probX = 0.999;		
					if ( probX < 0.001):
						probX = 0.001;
					#print i, probSum, probX, targetTest[i]
					#print target[i]*log(probX), (1-target[i])*log(1-probX)
					probSum += int(targetTest[i])*log(probX)+(1-int(targetTest[i]))*log(1-probX)
					if ( math.fabs(probX - int(targetTest[i])) > 0.5 ):
						totalOffByHalf = totalOffByHalf + 1		
				
					if ( int(targetTest[i]) == 1 ):
						totalPositive = totalPositive + 1
					if ( int(targetTest[i]) == 1 and probX < 0.5):
						totalPositiveOffByHalf = totalPositiveOffByHalf + 1
					if (probX > 0.5):
						totalPositivePredictions = totalPositivePredictions + 1			
				
				print "Total Off By > 0.5 ", totalOffByHalf
				print "Total Positive ", totalPositive
				print "Total Positive Off By Half ", totalPositiveOffByHalf
				print "Total Positive Predictions ", totalPositivePredictions
				print -probSum/len(prob)
		
	 
				avg += 	(-probSum/len(prob))/NumFolds

				predicted_probs = clf.predict_proba(testNew)  # was test						
				#print [x[1] for x in predicted_probs]
				predicted_list.append([x[1] for x in predicted_probs])
			
				dataset_blend_test_j[:, foldCount] = predicted_probs[:,1]
			
				foldCount = foldCount + 1
			
							
			dataset_blend_test[:,ExecutionIndex] = dataset_blend_test_j.mean(1)
			#now = datetime.datetime.now()
			#csv_io.write_delimited_file_GUID("../Submissions/stack_avg" + now.strftime("%Y%m%d%H%M") + "_" + str(avg) + ".csv", "PreProcessData/test_PatientGuid.csv", dataset_blend_test_j.mean(1))
			
			print "------------------------------------------------Average: ", avg
			open("stack_gb_data.txt","a").write(str(n_est)+','+str(learn_r)+','+str(avg)+"\n")
	
			avg_list = np.zeros(len(test))
			med_list = np.zeros(len(test))
		
			# For N folds, get the average/median for each prediction item in test set.
			for p in range(0, len(test)):
				temp_list =[]	
				for q in range(0, len(predicted_list)):		
					temp_list.append(  predicted_list[q][p]) 
				
				avg_list[p] = mean(temp_list) 
				med_list[p] = getMedian(temp_list) 
			
				#print p, q, temp_list, mean(temp_list), getMedian(temp_list)
				

			bootstrapLists.append(avg_list)

			ExecutionIndex = ExecutionIndex + 1
		
	# This would be used if we ran multiple runs with different training values.
	# Primitive stacking, should rather save data, and do formal stacking.
	if ( len(bootstrapLists) > 1 ):
		finalList = []
		for p in range(0, len(test)):
			temp_list =[]	
			for q in range(0, len(bootstrapLists)):		
				temp_list.append(  bootstrapLists[q][p]) 
			
			finalList.append( meanSpan(temp_list, spanDistance) )
		
			#print p, q, temp_list, meanSpan(temp_list, spanDistance)
	else:
		finalList = bootstrapLists[0]		
		
	#finalList = SimpleScale(finalList)
	avg_values = ["%f" % x for x in finalList]
	csv_io.write_delimited_file_GUID("../Submissions/gb_5fold_avg.csv", "PreProcessData/test_PatientGuid.csv", avg_values)	
	
	#for rec in dataset_blend_train:
	#	print rec
	
	return dataset_blend_train, dataset_blend_test
	#var = raw_input("Enter to terminate.")								
	
	
if __name__=="__main__":
	run_stack(448)
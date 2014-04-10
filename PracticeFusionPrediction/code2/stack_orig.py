#!/usr/bin/env python

from sklearn import svm
import csv_io
import math
from math import log
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import numpy as np

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
	
def run_stack():

	trainBase = csv_io.read_data("PreProcessData/training_PreProcess2.csv", False)
	test = csv_io.read_data("PreProcessData/test_PreProcess2.csv", False)
	
	avg = 0
	NumFolds = 5 # should be odd for median

	predicted_list = []
	
	spanDistance = 12
	bootstrapLists = []
	
	clfs = [RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=1, criterion='entropy'),
            GradientBoostingClassifier(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]
		

	
	print len(trainBase), len(test)
	dataset_blend_train = np.zeros((len(trainBase), len(clfs)))
	dataset_blend_test = np.zeros((len(test), len(clfs)))
	
	for ExecutionIndex, clf in enumerate(clfs):
		print clf
		
		predicted_list = []
		avg = 0
		
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

			clf.fit(train, target)
			prob = clf.predict_proba(trainTest) 
			
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

			predicted_probs = clf.predict_proba(test)  # was test						
			#print [x[1] for x in predicted_probs]
			predicted_list.append([x[1] for x in predicted_probs])
		
			dataset_blend_test_j[:, foldCount] = predicted_probs[:,1]
		
			foldCount = foldCount + 1
		
		dataset_blend_test[:,ExecutionIndex] = dataset_blend_test_j.mean(1)
		
		
		print "------------------------Average: ", avg

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
	csv_io.write_delimited_file_GUID("../Submissions/rf2_5fold_avg.csv", "PreProcessData/test_PatientGuid.csv", avg_values)	
	
	#for rec in dataset_blend_train:
	#	print rec
	
	return dataset_blend_train, dataset_blend_test
	#var = raw_input("Enter to terminate.")								
	
	
if __name__=="__main__":
	run_stack()
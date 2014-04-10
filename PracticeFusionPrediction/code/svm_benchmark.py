#!/usr/bin/env python

from sklearn import svm
import csv_io
import math
from math import log
from sklearn import cross_validation



def SimpleScale(probArray, floor = 0.001, ceiling = 0.999):

	minProb = 0.5
	maxProb = 0.5
	
	# search for min and max probs
	for i in range(0, len(probArray)):
		probX = probArray[i][1] # [1]
		
		if ( probX  > maxProb ):
			maxProb = probX
		if ( probX  < minProb ):
			minProb = probX
			
	#scale those below 0.5 down to 0 and above 0.5 up to 1		
	for i in range(0, len(probArray)):
		probX = probArray[i][1] # [1]			
					
		if ( probX < 0.5 ):
			probArray[i][1] = 0.5 - ((0.5 - probX)/(0.5 - minProb)) * 0.5 
			#print probX, probArray[i][1]
			
		if ( probX > 0.5 ):
			probArray[i][1] = 0.5 + ((probX - 0.5)/(maxProb - 0.5)) * 0.5 
			print probX, probArray[i][1]
		
		if ( probArray[i][1] > ceiling):
			probArray[i][1] = ceiling;		
		if ( probArray[i][1] < floor):
			probArray[i][1] = floor;
		
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
	
def main():

	trainBase = csv_io.read_data("PreProcessData/PreProcess2.csv", False)
	
	avg = 0
	NumFolds = 5 # should be odd for median

	predicted_list = []
	
	spanDistance = 12
	bootstrapLists = []
	
	
	CgList = [[0.0, -5.5]]
	

	for Cg in CgList:
		
		predicted_list = []

		Folds = cross_validation.KFold(len(trainBase) - 1, k=NumFolds, indices=True, shuffle=False, random_state=None)
		for train_index, test_index in Folds:

			trainBaseTemp = [trainBase[i+1] for i in train_index]
			#trainBaseTemp = trainBase
			target = [x[0] for x in trainBaseTemp]
			train = [x[1:] for x in trainBaseTemp]
	
			testBaseTemp = [trainBase[i+1] for i in test_index]
			#testBaseTemp = trainBase
			targetTest = [x[0] for x in testBaseTemp]
			trainTest = [x[1:] for x in testBaseTemp]
	
	
			test = csv_io.read_data("PreProcessData/PreTestData2.csv", False)
			test = [x[0:] for x in test]
	

			svc = svm.SVC(probability=True, C=10**Cg[0], gamma=2**Cg[1], cache_size=800, coef0=0.0, degree=3, kernel='rbf', shrinking=True, tol=0.001)
			
			svc.fit(train, target)
			prob = svc.predict_proba(trainTest) 
	
			prob = SimpleScale(prob) # scale output probababilities
	
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
			print "C/g: ", Cg[0], Cg[1]
			print -probSum/len(prob)
	
 
			avg += 	(-probSum/len(prob))/NumFolds

			predicted_probs = svc.predict_proba(test)  # was test
						
			prob = SimpleScale(prob) # scale output probababilities
						
			predicted_list.append([x[1] for x in predicted_probs])
				


		avg_list = []
		med_list = []
	
		# For N folds, get the average/median for each prediction item in test set.
		for p in range(0, len(test)):
			temp_list =[]	
			for q in range(0, len(predicted_list)):		
				temp_list.append(  predicted_list[q][p]) 
			
			avg_list.append( mean(temp_list) )
			med_list.append( getMedian(temp_list) )
		
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
		
			print p, q, temp_list, meanSpan(temp_list, spanDistance)
	else:
		finalList = bootstrapLists[0]		
		
		
	avg_values = ["%f" % x for x in finalList]
	csv_io.write_delimited_file("../Submissions/rf2_stack_avg.csv", avg_values)	
	
	
	print "Average: ", avg
		
	var = raw_input("Enter to terminate.")								
								
if __name__=="__main__":
	main()
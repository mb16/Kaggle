#!/usr/bin/env python

from sklearn import svm
import csv_io
import math
from math import log
from sklearn import cross_validation
from sklearn.ensemble import ExtraTreesClassifier


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

    #random.seed(5)
    #random.random()
	
    startCol = 0
    endCol = 1775  # max = 1775

    trainBase = csv_io.read_data("../Data/train.csv")
	
    result = 100
    avg = 0
    bootstraps = 9 # should be odd for median
	
    rnd_start = 456
	

    predicted_list = []
	
    spanDistance = 12
    bootstrapLists = []
	

    if ( True):
        
        predicted_list = []
        bs = cross_validation.Bootstrap(len(trainBase) - 1, n_bootstraps=bootstraps, train_size=0.7, random_state=0)
        for train_index, test_index in bs:

            trainBaseTemp = [trainBase[i+1] for i in train_index]
            #trainBaseTemp = trainBase
            target = [x[0] for x in trainBaseTemp]#[1001:3700]
            train = [x[1:] for x in trainBaseTemp]#[1001:3700]
	
            testBaseTemp = [trainBase[i+1] for i in test_index]
            #testBaseTemp = trainBase
            targetTest = [x[0] for x in testBaseTemp]#[1:1000]
            trainTest = [x[1:] for x in testBaseTemp]#[1:1000]
	
	
            test = csv_io.read_data("../Data/test.csv")
            test = [x[0:] for x in test]
	
	
            fo = open("rf_stats.txt", "a+")
    
	
            rf = ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=None, min_samples_split=1, min_samples_leaf=1, min_density=0.10000000000000001, max_features='auto', bootstrap=False, compute_importances=False, oob_score=False, n_jobs=1, random_state=None, verbose=0)

            rf.fit(train, target)
            prob = rf.predict_proba(trainTest)  # was test
	
            probSum = 0
            totalOffByHalf = 0
	
            for i in range(0, len(prob)):
                probX = prob[i][1] # [1]
                if ( probX > 0.999999999999):
                    probX = 0.999999999999;		
                if ( probX < 0.000000000001):
                    probX = 0.000000000001;
                #print i, probSum, probX, target[i]
                #print target[i]*log(probX), (1-target[i])*log(1-probX)
                probSum += targetTest[i]*log(probX)+(1-targetTest[i])*log(1-probX)
                if ( math.fabs(probX - targetTest[i]) > 0.5 ):
                    totalOffByHalf = totalOffByHalf + 1		
			
            print "Total Off By > 0.5 ", totalOffByHalf
            print -probSum/len(prob)
	
            #fo.write(str(C) + "," + str(g) + "," + str(-probSum/len(prob)));
	
            avg += 	(-probSum/len(prob))/bootstraps

            predicted_probs = rf.predict_proba(test)  # was test
            predicted_list.append([x[1] for x in predicted_probs])
	
	
            fo.close()


        avg_list = []
        med_list = []
	
	
        for p in range(0, len(test)):
            temp_list =[]	
            for q in range(0, len(predicted_list)):		
		        temp_list.append(  predicted_list[q][p]) 
			
            avg_list.append( mean(temp_list) )
            med_list.append( getMedian(temp_list) )
		
            print p, q, temp_list, mean(temp_list), getMedian(temp_list)
		
        bootstrapLists.append(avg_list)
		
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
    csv_io.write_delimited_file("../Submissions/et_stack_avg_benchmark.csv", avg_values)	
	
	
    print "Average: ", avg
		
    var = raw_input("Enter to terminate.")								
								
if __name__=="__main__":
    main()
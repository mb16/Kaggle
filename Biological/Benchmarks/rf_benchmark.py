#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import GradientBoostingClassifier # not in version 0.1, but will be in 0.11
from sklearn import ensemble
from sklearn import neighbors, datasets
import csv_io
import math
from math import log

def main():

	# this method does not seem to benefit from using less than all columns of data.
    startCol = 0
    endCol = 1775  # max = 1775

    train = csv_io.read_data("../Data/train.csv")
    #target = [x[0] for x in train][1:3000]
    #targetTest = [x[0] for x in train][3001:]
    #trainTest = [x[startCol+1:endCol+1] for x in train][3001:]
    #test = csv_io.read_data("../Data/test.csv")
    #test = [x[startCol:endCol] for x in test]
	
    #train = [x[startCol+1:endCol+1] for x in train][1:3000]	
	
	#trainBaseTemp = [trainBase[i+1] for i in train_index]
    trainBaseTemp = train
    target = [x[0] for x in trainBaseTemp][1001:3700]
    train = [x[1:] for x in trainBaseTemp][1001:3700]
	
    #testBaseTemp = [trainBase[i+1] for i in test_index]
    #testBaseTemp = train
    targetTest = [x[0] for x in trainBaseTemp][1:1000]
    trainTest = [x[1:] for x in trainBaseTemp][1:1000]
	
	
    test = csv_io.read_data("../Data/test.csv")
    #test = [x for x in test]
	
	
    fo = open("rf_stats.txt", "a+")

	
    #rf = RandomForestClassifier(n_estimators=200, min_density=0.2, criterion="gini", random_state=685) # , max_features=None
    rf = ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=None, min_samples_split=1, min_samples_leaf=1, min_density=0.10000000000000001, max_features='auto', bootstrap=False, compute_importances=False, oob_score=False, n_jobs=1, random_state=None, verbose=0)
    #rf = ensemble.GradientBoostingClassifier(n_estimators=100, learn_rate=0.1) # max_depth=1, random_state=0
    #rf = neighbors.KNeighborsClassifier(n_neighbors=150, weights='distance') # 'distance'
	
    rf.fit(train, target)
    #prob = rf.predict(test) # changed from test
    prob = rf.predict_proba(trainTest) # changed from test

    #prob = ["%f" % x[1] for x in prob]
    #prob = ["%f" % x for x in prob]

    result = 100
    probSum = 0
    for i in range(0, len(prob)):
        probX = prob[i][1] # [1]
        if ( probX > 0.999999999999):
            probX = 0.999999999999;		
        if ( probX < 0.000000000001):
            probX = 0.000000000001;
        print i, probSum, probX, target[i]
        print target[i]*log(probX), (1-target[i])*log(1-probX)
        probSum += targetTest[i]*log(probX)+(1-targetTest[i])*log(1-probX)
	
        #print probSum	
        #print len(prob)	
        #print "C: ", 10**C, " gamma: " ,2**g
        print -probSum/len(prob)
	

	
    if ( -probSum/len(prob) < result ):
        result = -probSum/len(prob)
        predicted_probs = rf.predict_proba(test)  # was test
        predicted_probs = ["%f" % x[1] for x in predicted_probs]
        csv_io.write_delimited_file("../Submissions/svm_benchmark.csv", predicted_probs)
        print "Generated Data!!"
		
    fo.write(str(5) + str(5)+ str(5));
		
    fo.close()
		
    #csv_io.write_delimited_file("../Submissions/rf_benchmark_test2.csv", predicted_probs)

    #predicted_probs = rf.predict_proba(train) # changed from test
 
    #predicted_probs = ["%f" % x[1] for x in predicted_probs]
    #predicted_probs = rf.predict(train) # changed from test
    #predicted_probs = ["%f" % x for x in predicted_probs]	
	
    #csv_io.write_delimited_file("../Submissions/rf_benchmark_train2.csv", predicted_probs)
	
	
    var = raw_input("Enter to terminate.")								
								
if __name__=="__main__":
    main()
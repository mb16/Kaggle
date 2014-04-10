#!/usr/bin/env python


from sklearn.linear_model import RidgeClassifier
import csv_io
import math
from math import log

def main():

    startCol = 0
    endCol = 50  # max = 1775

    train = csv_io.read_data("../Data/train.csv")
    target = [x[0] for x in train][1:3000]
    targetTest = [x[0] for x in train][3001:]
    trainTest = [x[startCol+1:endCol+1] for x in train][3001:]
    test = csv_io.read_data("../Data/test.csv")
    test = [x[startCol:endCol] for x in test]
	
    train = [x[startCol+1:endCol+1] for x in train][1:3000]	
	
    fo = open("knn_stats.txt", "a+")

    rf = RidgeClassifier(alpha=0.01, fit_intercept=True, normalize=False, copy_X=True, tol=0.001) 
	
    rf.fit(train, target)
    prob = rf.predict(trainTest) # changed from test


    result = 100
    probSum = 0
    for i in range(0, len(prob)):
        probX = prob[i] # [1]
        if ( probX > 0.7):
            probX = 0.7;		
        if ( probX < 0.3):
            probX = 0.3;
        print i, probSum, probX, target[i]
        print target[i]*log(probX), (1-target[i])*log(1-probX)
        probSum += targetTest[i]*log(probX)+(1-targetTest[i])*log(1-probX)
	
        #print probSum	
        #print len(prob)	
        #print "C: ", 10**C, " gamma: " ,2**g
        print -probSum/len(prob)
	

	
    if ( -probSum/len(prob) < result ):
        result = -probSum/len(prob)
        predicted_probs = rf.predict(test)  # was test
        predicted_probs = ["%f" % x for x in predicted_probs]
        csv_io.write_delimited_file("../Submissions/knn.csv", predicted_probs)
        print "Generated Data!!"
		
    #fo.write(str(5) + str(5)+ str(5));
		
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
#!/usr/bin/env python


from sklearn.linear_model import SGDClassifier
import csv_io
import math
from math import log

def main():

    startCol = 0
    endCol = 1775  # max = 1775

    train = csv_io.read_data("../Data/train.csv")
    target = [x[0] for x in train][1:3000]
    targetTest = [x[0] for x in train][3001:]
    trainTest = [x[startCol+1:endCol+1] for x in train][3001:]
    test = csv_io.read_data("../Data/test.csv")
    test = [x[startCol:endCol] for x in test]
	
    train = [x[startCol+1:endCol+1] for x in train][1:3000]	
	
    fo = open("knn_stats.txt", "a+")

    rf = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, rho=0.84999999999999998, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, n_jobs=1, seed=0, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None) 
	
    rf.fit(train, target)
    prob = rf.predict_proba(trainTest) # changed from test
    print prob

    result = 100
    probSum = 0
    for i in range(0, len(prob)):
        probX = prob[i] # [1]
        if ( probX > 0.9):
            probX = 0.9;		
        if ( probX < 0.1):
            probX = 0.1;
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
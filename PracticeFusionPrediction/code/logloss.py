#!/usr/bin/env python

import csv_io
from math import log

def main():
    train = csv_io.read_data("../Data/train.csv")
    target = [x[0] for x in train][1:] # skip the headers
    probabilities = csv_io.read_data("../Submissions/svm_benchmark.csv")
    prob = [x[0] for x in probabilities]
	
    probSum = 0
	
    for i in range(0, len(prob)):
        #tempProb = max(prob[i], 0.000001)
        #tempProb = min(tempProb, 0.999999)
        #tempProb = max(prob[i], 0.1)
        #tempProb = min(tempProb, 0.9)
        print i, probSum, prob[i], target[i]
        print target[i]*log(prob[i]), (1-target[i])*log(1-prob[i])
        probSum += target[i]*log(prob[i])+(1-target[i])*log(1-prob[i])
	
    print probSum	
    print len(prob)	
    print -probSum/len(prob)
	#result = (-1/len(probs))*mySum;

    var = raw_input("Enter to terminate.")

if __name__=="__main__":
    main()

#LogLoss<-function(actual, predicted)
#{
#result<- -1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
#return(result)
#}
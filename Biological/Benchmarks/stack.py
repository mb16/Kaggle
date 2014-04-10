#!/usr/bin/env python

import csv_io
from math import log
import math


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

    et = csv_io.read_data("../Submissions/et_stack_avg_benchmark.csv", False)
    rbf = csv_io.read_data("../Submissions/svm-rbf-bootstrap-stack_meanSpan_benchmark.csv", False)
    poly = csv_io.read_data("../Submissions/svm-poly-bootstrap-stack_meanSpan_benchmark.csv", False)
    rf = csv_io.read_data("../Submissions/rf2_avg_benchmark.csv", False)
    gb = csv_io.read_data("../Submissions/gb_avg_benchmark.csv", False)

    stack = []
    stack.append(et)
    stack.append(rbf)
    stack.append(poly)
    stack.append(rf)
    stack.append(gb)	
	
    spanDistance = 3
    finalList = []
    for p in range(0, len(stack[0])):
        temp_list =[]	
        for q in range(0, len(stack)):		
		    temp_list.append( stack[q][p][0]) 

        avg = sum(temp_list)/float(len(stack))	

        if ( avg < 0.5 ):
            finalList.append(0.2) 
            #finalList.append(min(temp_list)) 
            print p, q, temp_list, avg, min(temp_list)
        else:		
            finalList.append(0.80) 
		    #finalList.append(max(temp_list)) 
            print p, q, temp_list, avg, max(temp_list)
			
        #finalList.append( meanSpan(temp_list, spanDistance) )
        #print p, q, temp_list, meanSpan(temp_list, spanDistance)
  			
		
    finalStack = ["%f" % x for x in finalList]
    csv_io.write_delimited_file("../Submissions/stack.csv", finalStack)	
	
	

    var = raw_input("Enter to terminate.")

if __name__=="__main__":
	main()


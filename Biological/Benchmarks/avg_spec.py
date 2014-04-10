#!/usr/bin/env python

import csv_io
from math import log
import math

def main():

	a = [2.0,4.0,14.0,11.0, 9.0]
	a = sorted(a)
	N = 3
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
		
	for j in range(minSpanIndex,minSpanIndex + N):	
		spanSum = spanSum + a[j]		
		#print "sum: ", j, a[j], spanSum
		
	#print span
	#print spanSum
	#print spanSum/float(N)
	#print minSpan, minSpanIndex
		    


	var = raw_input("Enter to terminate.")

if __name__=="__main__":
	main()


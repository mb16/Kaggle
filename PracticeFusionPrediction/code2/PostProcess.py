#!/usr/bin/env python


import string

from sklearn import svm
import csv_io
import math
from math import log
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import scipy

import datetime

def toFloat(str):
	return float(str)

def SimpleScale(probArray, floor = 0.001, ceiling = 0.999):

	minProb = 0.5
	maxProb = 0.5
	
	# search for min and max probs
	for i in range(0, len(probArray)):
		probX = toFloat(probArray[i][1]) # [1]
		
		if ( probX  > maxProb ):
			maxProb = probX
		if ( probX  < minProb ):
			minProb = probX
			
	#scale those below 0.5 down to 0 and above 0.5 up to 1		
	for i in range(0, len(probArray)):
		probX = toFloat(probArray[i][1]) # [1]			
					
		if ( probX < 0.5 ):
			probArray[i][1] = 0.5 - ((0.5 - probX)/(0.5 - minProb)) * 0.5 
			#print probX, probArray[i]
			
		if ( probX > 0.5 ):
			probArray[i][1] = 0.5 + ((probX - 0.5)/(maxProb - 0.5)) * 0.5 
			#probArray[i] = ceiling;		
			
		if ( probArray[i][1] < floor):
			probArray[i][1] = floor;

		if ( probArray[i][1] > ceiling):
			probArray[i][1] = ceiling;
			
	print "SimpleScale: ", minProb, maxProb
	
	return probArray		
	

def PreProcess3():
	filename = "stack201208301510"

	data = csv_io.read_data("../Submissions/" + filename + ".csv", False)
	data = SimpleScale(data, floor = 0.05, ceiling = 0.90)  # took 0.389 score an lowered to 0.40, not good...

	csv_io.write_delimited_file("../Submissions/" + filename + "_SimpleScale.csv", data)	

	

			
								
if __name__=="__main__":
	PreProcess3()
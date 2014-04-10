#!/usr/bin/env python

import csv_io
import string

import numpy as np
import datetime

import os
import shutil
import math

def Analyze1():

	Threshold = 4.0  
	targetFile = "Target_Stack_20121017110223_3.06649134025_GradientBoos.csv"

	trainBase = csv_io.read_data("PreProcessData/training_PreProcess3.csv", skipFirstLine = False, split = "\t")
	shutil.copy2("PreProcessData/test_PreProcess3.csv", "PreProcessData/test_PreProcess8.csv")	
	shutil.copy2("PreProcessData/DataClassList3.csv", "PreProcessData/DataClassList8.csv")	
	weights = csv_io.read_data("PreProcessData/Weights.csv", skipFirstLine = False)
	
	target = [x[0] for x in trainBase]
	
	
	print "Loading Data"
	trainNew = []
	
	probSum = 0.0
	weightSum = 0
	
	trn = csv_io.read_data("../predictions/" + targetFile, split="," ,skipFirstLine = False)
	for row, datum in enumerate(trn):

		if ( abs(datum[0] - target[row]) > Threshold):
			print datum[0], target[row]
			trainNew.append(trainBase[row])
			
			probSum += weights[row][0] * math.fabs(target[row] - datum[0])
			weightSum += weights[row][0]
		
		
	print "Train Score: ", (probSum/weightSum)	
	print len(trainNew)
	csv_io.write_delimited_file("PreProcessData/training_PreProcess8" + ".csv", trainNew, delimiter="\t")	
	
											
if __name__=="__main__":
	Analyze1()
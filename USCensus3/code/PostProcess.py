#!/usr/bin/env python

import csv_io
import string

import numpy as np
import datetime

import os

def PostProcess():

	lossThreshold = 4.0  # best seems to be about 4.0
	model = "Long-Lat KNN5"

	#used only for targets values.
	trainBase = csv_io.read_data("PreProcessData/training_PreProcess4_40.csv", skipFirstLine = False, split = "\t")
	test = csv_io.read_data("PreProcessData/test_PreProcess4_40.csv", False)
	weights = csv_io.read_data("PreProcessData/Weights.csv", skipFirstLine = False)

	target = [x[0] for x in trainBase]
	
	
	
	
	stackFiles = []
	for filename in os.listdir("../predictions"):
		parts = filename.split("_")
		if ( filename[0:5] == "Stack" and float(parts[2]) < lossThreshold):

			stackFiles.append(filename)
	
	
	dataset_blend_train = np.zeros((len(trainBase), len(stackFiles)))
	dataset_blend_test = np.zeros((len(test), len(stackFiles)))
	
	print "Loading Data"
	for fileNum, file in enumerate(stackFiles):
		print file
		trn = csv_io.read_data("../predictions/Target_" + file, split="," ,skipFirstLine = False)
		for row, datum in enumerate(trn):
			dataset_blend_train[row, fileNum] = datum[0]
		
		tst = csv_io.read_data("../predictions/" + file, split="," ,skipFirstLine = False)
		for row, datum in enumerate(tst):
			dataset_blend_test[row, fileNum] = datum[0]

	np.savetxt('temp/dataset_blend_trainX.txt', dataset_blend_train)
	np.savetxt('temp/dataset_blend_testX.txt', dataset_blend_test)
	np.savetxt('temp/dataset_blend_trainY.txt', target)
	print "Num file processed: ", len(stackFiles), "Threshold: ", lossThreshold

	
	
											
if __name__=="__main__":
	PostProcess()
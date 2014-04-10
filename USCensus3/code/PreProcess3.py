#!/usr/bin/env python

import csv_io
import math

from math import log
import string
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

import shutil

def toFloat(str):
	return float(str)

def myFunc(arr):
	#print "myFunc: ", arr
	arr1 = arr[0]
	
	newArr = []
	for index, data in enumerate(arr1):
		if index == 0:
			newArr.append(0.0)
		else:
			#newArr.append(1.0/data)
			newArr.append(1.0)
	#print newArr
	return [newArr]
	
	
def PreProcess3():


	trainBase = csv_io.read_data("PreProcessData/training_PreProcess2.csv", split="\t" ,skipFirstLine = False)
	test = csv_io.read_data("PreProcessData/test_PreProcess2.csv", split="\t" ,skipFirstLine = False)
	weights = csv_io.read_data("PreProcessData/Weights.csv", skipFirstLine = False)
	
	print "Train Size: ", len(trainBase[0]), "Test Size: ", len(test[0])

	
	shutil.copy2("PreProcessData/DataClassList2.csv", "PreProcessData/DataClassList3.csv")
	
	lat = len(trainBase[0]) - 2
	long = len(trainBase[0]) - 1

	
	target = [x[0] for x in trainBase]
	train = [x[lat:long + 1] for x in trainBase]

	
	n_neighborsArr = [5]
	leaf_sizeArr = [30]
	for n_neighbor in n_neighborsArr:
		for leaf_s in leaf_sizeArr:	
		
			print "Training neighbors: ", n_neighbor, "leaf_size: ", leaf_s
			
			neigh = KNeighborsRegressor(n_neighbors=n_neighbor,warn_on_equidistant=False, leaf_size=leaf_s, algorithm="ball_tree", weights=myFunc) 
			neigh.fit(train, target) 
						
			probSum = 0
			weightSum = 0
			
			for index, data in enumerate(trainBase):
				pred = neigh.predict([data[lat], data[long]])
				#print data[lat], data[long], "Prediction: ", pred[0], "Target: ", target[index]
				if ( len(n_neighborsArr) == 1 ):
					trainBase[index].append(pred[0])

				probSum += weights[index][0] * math.fabs(target[index] - pred[0])
				weightSum += weights[index][0] 
			
			
			print "Score: ", probSum/weightSum
			if ( len(n_neighborsArr) > 1 ):	
				continue
			
			for index, data in enumerate(test):
				pred = neigh.predict([data[lat - 1], data[long - 1]])
				#print data[lat - 1], data[long - 1], "Prediction: ", pred[0]
				if ( len(n_neighborsArr) == 1 ):
					test[index].append(pred[0])
			
	
	
	if ( len(n_neighborsArr) > 1 ):	
		return

	
	with open("PreProcessData/DataClassList3.csv", "a") as myfile:
		myfile.write("Lat-Long-Predictor\n")

	print "Writing Data"
	csv_io.write_delimited_file("PreProcessData/training_PreProcess3.csv", trainBase, delimiter="\t")		
	csv_io.write_delimited_file("PreProcessData/test_PreProcess3.csv", test, delimiter="\t")
	print "Done."	


		
if __name__=="__main__":


	


	print "Creating Lat/Long predictors."
	PreProcess3()

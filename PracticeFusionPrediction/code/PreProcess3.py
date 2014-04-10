#!/usr/bin/env python

import csv_io
import math

from math import log
import string
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def toFloat(str):
	return float(str)


def PreProcess2():


	trainBase = csv_io.read_data("PreProcessData/training_PreProcess2_temp.csv", False)
	test = csv_io.read_data("PreProcessData/test_PreProcess2_temp.csv", False)

	target = [x[0] for x in trainBase]
	train = [x[1:] for x in trainBase]
	
	IndexList = [2,3,4,5,6]
	
	for myIndex in IndexList:
		
		
		MTrain = []
		FTrain = []
		MTarget = []
		FTarget = []
		
		for index, data in enumerate(train):
			if ( data[0] == "0" ):
				MTrain.append([data[1], data[myIndex]])
				MTarget.append(target[index])
				#print "M", data[1], data[myIndex]
			if ( data[0] == "1" ):
				FTrain.append([data[1], data[myIndex]])	
				FTarget.append(target[index])
				#print "F", data[1], data[myIndex]
				
		#print MTrain			
		print len(MTrain), len(FTrain),len(MTarget), len(FTarget)
		#Mneigh = KNeighborsClassifier(n_neighbors=6, algorithm='kd_tree') # not good
		#Fneigh = KNeighborsClassifier(n_neighbors=6, algorithm='kd_tree')
		
		Mneigh = RandomForestClassifier()
		Fneigh = RandomForestClassifier()

		#Mneigh = GradientBoostingClassifier()  # better, but not as good as RF
		#Fneigh = GradientBoostingClassifier()
		
		Mneigh.fit(MTrain, MTarget) 
		Fneigh.fit(FTrain, FTarget) 
		

		count = 0
		for index, data in enumerate(train):
			if ( data[0] == "0" ):
				pred = Mneigh.predict_proba([data[1], data[myIndex]])
				#print "M", data[1], data[myIndex], pred[0][1], target[index]
				trainBase[index].append(pred[0][1])
				if ( str(pred[0][1]) == str(target[index])):
					count = count + 1
			if ( data[0] == "1" ):
				pred = Fneigh.predict_proba([data[1], data[myIndex]])
				#print "F", data[1], data[myIndex], pred[0][1], target[index]
				trainBase[index].append(pred[0][1])
				if ( str(pred[0][1]) == str(target[index])):
					count = count + 1


		
		for index, data in enumerate(test):
			if ( data[0] == "0" ):
				pred = Mneigh.predict_proba([data[1], data[myIndex]])
				#print "M", data[1], data[myIndex], pred[0][1], target[index]
				test[index].append(pred[0][1])
			if ( data[0] == "1" ):
				pred = Fneigh.predict_proba([data[1], data[myIndex]])
				#print "F", data[1], data[myIndex], pred[0][1], target[index]
				test[index].append(pred[0][1])	
		
		csv_io.write_delimited_file("PreProcessData/training_PreProcess2_temp_a.csv", trainBase)		
		csv_io.write_delimited_file("PreProcessData/test_PreProcess2_temp_a.csv", test)
	
		print myIndex, count	
	
	
	with open("PreProcessData/DataClassList.csv", "a") as myfile:
		myfile.write("RF_Gender_Age_height\n")
		myfile.write("RF_Gender_Age_weight\n")
		myfile.write("RF_Gender_Age_BMI\n")
		myfile.write("RF_Gender_Age_systolicBP\n")
		myfile.write("RF_Gender_Age_diastolicBP\n")



		
if __name__=="__main__":
	PreProcess2()
#!/usr/bin/env python

import csv_io
import math

from math import log
import string
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def toFloat(str):
	return float(str)


def PreProcess3():


	trainBase = csv_io.read_data("PreProcessData/training_PreProcess2.csv", False)
	test = csv_io.read_data("PreProcessData/test_PreProcess2.csv", False)

	target = [x[0] for x in trainBase]
	train = [x[1:] for x in trainBase]
				
	
	DataClassList = csv_io.read_data("PreProcessData/DataClassList.csv", False)

	
	for myIndex in range(2,20):	# use for quick processing, and only training the most critical features
	#for myIndex in range(2,75): # train most critical features (longer processing)
	#for myIndex in range(2,len(train[0]) - 2): # train all features ( can take a long time)
		
		
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
				
		
		#print len(MTrain), len(FTrain),len(MTarget), len(FTarget)

		# better than GradBoost, and much better than KNN, should tune.
		#Mneigh = RandomForestClassifier() # severe overfitting
		#Fneigh = RandomForestClassifier()

		Mneigh = GradientBoostingClassifier() 
		Fneigh = GradientBoostingClassifier()		

		#Mneigh = LogisticRegression() 
		#Fneigh = LogisticRegression()	
		
		#Mneigh = KNeighborsClassifier(n_neighbors=25) 
		#Fneigh = KNeighborsClassifier(n_neighbors=25)	
		
		
		Mneigh.fit(MTrain, MTarget) 
		Fneigh.fit(FTrain, FTarget) 
		
		# for index, data in enumerate(train):
			# if ( data[0] == "0" ):
				# pred = Mneigh.predict_proba([data[1], data[myIndex]])
				#print "M", data[1], data[myIndex], pred[0][1], target[index]
				# trainBase[index].append(pred[0][1])
			# if ( data[0] == "1" ):
				# pred = Fneigh.predict_proba([data[1], data[myIndex]])
				#print "F", data[1], data[myIndex], pred[0][1], target[index]
				# trainBase[index].append(pred[0][1])


		
		# for index, data in enumerate(test):
			# if ( data[0] == "0" ):
				# pred = Mneigh.predict_proba([data[1], data[myIndex]])
				#print "M", data[1], data[myIndex], pred[0][1], target[index]
				# test[index].append(pred[0][1])
			# if ( data[0] == "1" ):
				# pred = Fneigh.predict_proba([data[1], data[myIndex]])
				#print "F", data[1], data[myIndex], pred[0][1], target[index]
				# test[index].append(pred[0][1])	
		
		

		for index, data in enumerate(train):
			if ( data[0] == "0" ):
				pred = Mneigh.predict_proba([data[1], data[myIndex]])
				#print "M", data[1], data[myIndex], pred[0][1], target[index]
				trainBase[index].append(pred[0][1])
			if ( data[0] == "1" ):
				pred = Fneigh.predict_proba([data[1], data[myIndex]])
				#print "F", data[1], data[myIndex], pred[0][1], target[index]
				trainBase[index].append(pred[0][1])


		
		for index, data in enumerate(test):
			if ( data[0] == "0" ):
				pred = Mneigh.predict_proba([data[1], data[myIndex]])
				#print "M", data[1], data[myIndex], pred[0][1], target[index]
				test[index].append(pred[0][1])
			if ( data[0] == "1" ):
				pred = Fneigh.predict_proba([data[1], data[myIndex]])
				#print "F", data[1], data[myIndex], pred[0][1], target[index]
				test[index].append(pred[0][1])	
		

		print "Processing Feature Index: ", myIndex, " of ", len(train[0])	
	
	
		with open("PreProcessData/DataClassList.csv", "a") as myfile:
			myfile.write("NEW_Gender-Age-Class_" + str(DataClassList[myIndex][0]) + "_" + str(myIndex) + "\n")

	print "Writing Data"
	csv_io.write_delimited_file("PreProcessData/training_PreProcess3_TEMP.csv", trainBase)		
	csv_io.write_delimited_file("PreProcessData/test_PreProcess3_TEMP.csv", test)
	print "Done."	


def PreProcess3a():


	trainBase = csv_io.read_data("PreProcessData/training_PreProcess3_TEMP.csv", False)
	test = csv_io.read_data("PreProcessData/test_PreProcess3_TEMP.csv", False)

	target = [x[0] for x in trainBase]
	train = [x[1:] for x in trainBase]
				
	
	DataClassList = csv_io.read_data("PreProcessData/DataClassList.csv", False)

	
	for myIndex in range(685,1019):	# use for quick processing, and only training the most critical features
	#for myIndex in range(2,75): # train most critical features (longer processing)
	#for myIndex in range(2,len(train[0]) - 2): # train all features ( can take a long time)
		
		
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
				
		
		#print len(MTrain), len(FTrain),len(MTarget), len(FTarget)

		# better than GradBoost, and much better than KNN, should tune.
		#Mneigh = RandomForestClassifier() # severe overfitting
		#Fneigh = RandomForestClassifier()

		Mneigh = GradientBoostingClassifier() 
		Fneigh = GradientBoostingClassifier()		

		#Mneigh = KNeighborsClassifier(n_neighbors=25) 
		#Fneigh = KNeighborsClassifier(n_neighbors=25)	
		
		
		Mneigh.fit(MTrain, MTarget) 
		Fneigh.fit(FTrain, FTarget) 
		


		for index, data in enumerate(train):
			if ( data[0] == "0" ):
				pred = Mneigh.predict_proba([data[1], data[myIndex]])
				#print "M", data[1], data[myIndex], pred[0][1], target[index]
				trainBase[index].append(pred[0][1])
			if ( data[0] == "1" ):
				pred = Fneigh.predict_proba([data[1], data[myIndex]])
				#print "F", data[1], data[myIndex], pred[0][1], target[index]
				trainBase[index].append(pred[0][1])


		
		for index, data in enumerate(test):
			if ( data[0] == "0" ):
				pred = Mneigh.predict_proba([data[1], data[myIndex]])
				#print "M", data[1], data[myIndex], pred[0][1], target[index]
				test[index].append(pred[0][1])
			if ( data[0] == "1" ):
				pred = Fneigh.predict_proba([data[1], data[myIndex]])
				#print "F", data[1], data[myIndex], pred[0][1], target[index]
				test[index].append(pred[0][1])	
		

		print "Processing Feature Index: ", myIndex, " of ", len(train[0])	
	
	
		with open("PreProcessData/DataClassList.csv", "a") as myfile:
			myfile.write("NEW_Gender-Age-Class_" + str(DataClassList[myIndex][0]) + "_" + str(myIndex) + "\n")

	print "Writing Data"
	csv_io.write_delimited_file("PreProcessData/training_PreProcess3.csv", trainBase)		
	csv_io.write_delimited_file("PreProcessData/test_PreProcess3.csv", test)
	print "Done."	

	
def PreProcess3x():


	trainBase = csv_io.read_data("PreProcessData/training_PreProcess2.csv", False)
	test = csv_io.read_data("PreProcessData/test_PreProcess2.csv", False)

	target = [x[0] for x in trainBase]
	train = [x[1:] for x in trainBase]
			
	trainBaseNew = []
	testNew = []
	
	DataClassList = csv_io.read_data("PreProcessData/DataClassList.csv", False)

	with open("PreProcessData/DataClassListLG.csv", "a") as myfile:
		myfile.write("NEW_Gender-Age-Class_" + str(DataClassList[0][0]) + "_" + str(0) + "\n")
		myfile.write("NEW_Gender-Age-Class_" + str(DataClassList[1][0]) + "_" + str(1) + "\n")
		
	for index, data in enumerate(train):
		trainBaseNew.append([data[0]])
		trainBaseNew.append([data[1]])

		
	for index, data in enumerate(test):
		testNew.append([data[0]])
		testNew.append([data[1]])	
		
	
	for myIndex in range(2, 1573):	# use for quick processing, and only training the most critical features
	#for myIndex in range(2,75): # train most critical features (longer processing)
	#for myIndex in range(2,len(train[0]) - 2): # train all features ( can take a long time)
		
		
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
				
		
		#print len(MTrain), len(FTrain),len(MTarget), len(FTarget)

		# better than GradBoost, and much better than KNN, should tune.
		#Mneigh = RandomForestClassifier() # severe overfitting
		#Fneigh = RandomForestClassifier()

		Mneigh = LogisticRegression() 
		Fneigh = LogisticRegression()		

		#Mneigh = KNeighborsClassifier(n_neighbors=25) 
		#Fneigh = KNeighborsClassifier(n_neighbors=25)	
		
		
		Mneigh.fit(MTrain, MTarget) 
		Fneigh.fit(FTrain, FTarget) 
		


		for index, data in enumerate(train):
			if ( data[0] == "0" ):
				pred = Mneigh.predict_proba([data[1], data[myIndex]])
				#print "M", data[1], data[myIndex], pred[0][1], target[index]
				trainBaseNew[index].append(pred[0][1])
			if ( data[0] == "1" ):
				pred = Fneigh.predict_proba([data[1], data[myIndex]])
				#print "F", data[1], data[myIndex], pred[0][1], target[index]
				trainBaseNew[index].append(pred[0][1])
			print len(trainBaseNew[index])

		
		for index, data in enumerate(test):
			if ( data[0] == "0" ):
				pred = Mneigh.predict_proba([data[1], data[myIndex]])
				#print "M", data[1], data[myIndex], pred[0][1], target[index]
				testNew[index].append(pred[0][1])
			if ( data[0] == "1" ):
				pred = Fneigh.predict_proba([data[1], data[myIndex]])
				#print "F", data[1], data[myIndex], pred[0][1], target[index]
				testNew[index].append(pred[0][1])	
			print len(testNew[index])

		print "Processing Feature Index: ", myIndex, " of ", len(train[0])	
	
	
		with open("PreProcessData/DataClassListLG.csv", "a") as myfile:
			myfile.write("NEW_Gender-Age-Class_" + str(DataClassList[myIndex][0]) + "_" + str(myIndex) + "\n")

	print "Writing Data"
	csv_io.write_delimited_file("PreProcessData/training_PreProcess3.csv", trainBaseNew)		
	csv_io.write_delimited_file("PreProcessData/test_PreProcess3.csv", testNew)
	print "Done."		

	
def PreProcess3b():


	trainBase = csv_io.read_data("PreProcessData/training_PreProcess2.csv", False)
	test = csv_io.read_data("PreProcessData/test_PreProcess2.csv", False)

	target = [x[0] for x in trainBase]
	train = [x[1:] for x in trainBase]
			
	trainBaseNew = []
	testNew = []
	
	DataClassList = csv_io.read_data("PreProcessData/DataClassList.csv", False)

	with open("PreProcessData/DataClassListLG.csv", "a") as myfile:
		myfile.write("NEW_Gender-Age-Class_" + str(DataClassList[0][0]) + "_" + str(0) + "\n")
		myfile.write("NEW_Gender-Age-Class_" + str(DataClassList[1][0]) + "_" + str(1) + "\n")
		
	for index, data in enumerate(train):
		trainBaseNew.append([data[0]])
		trainBaseNew.append([data[1]])

		
	for index, data in enumerate(test):
		testNew.append([data[0]])
		testNew.append([data[1]])	
		
	
	for myIndex in range(2, 1573):	# use for quick processing, and only training the most critical features
	#for myIndex in range(2,75): # train most critical features (longer processing)
	#for myIndex in range(2,len(train[0]) - 2): # train all features ( can take a long time)
		
		
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
				
		
		#print len(MTrain), len(FTrain),len(MTarget), len(FTarget)

		# better than GradBoost, and much better than KNN, should tune.
		#Mneigh = RandomForestClassifier() # severe overfitting
		#Fneigh = RandomForestClassifier()

		Mneigh = LogisticRegression() 
		Fneigh = LogisticRegression()		

		#Mneigh = KNeighborsClassifier(n_neighbors=25) 
		#Fneigh = KNeighborsClassifier(n_neighbors=25)	
		
		
		Mneigh.fit(MTrain, MTarget) 
		Fneigh.fit(FTrain, FTarget) 
		


		for index, data in enumerate(train):
			if ( data[0] == "0" ):
				pred = Mneigh.predict_proba([data[1], data[myIndex]])
				#print "M", data[1], data[myIndex], pred[0][1], target[index]
				trainBaseNew[index].append(pred[0][1])
			if ( data[0] == "1" ):
				pred = Fneigh.predict_proba([data[1], data[myIndex]])
				#print "F", data[1], data[myIndex], pred[0][1], target[index]
				trainBaseNew[index].append(pred[0][1])
			print len(trainBaseNew[index])

		
		for index, data in enumerate(test):
			if ( data[0] == "0" ):
				pred = Mneigh.predict_proba([data[1], data[myIndex]])
				#print "M", data[1], data[myIndex], pred[0][1], target[index]
				testNew[index].append(pred[0][1])
			if ( data[0] == "1" ):
				pred = Fneigh.predict_proba([data[1], data[myIndex]])
				#print "F", data[1], data[myIndex], pred[0][1], target[index]
				testNew[index].append(pred[0][1])	
			print len(testNew[index])

		print "Processing Feature Index: ", myIndex, " of ", len(train[0])	
	
	
		with open("PreProcessData/DataClassListLG.csv", "a") as myfile:
			myfile.write("NEW_Gender-Age-Class_" + str(DataClassList[myIndex][0]) + "_" + str(myIndex) + "\n")

	print "Writing Data"
	csv_io.write_delimited_file("PreProcessData/training_PreProcess3.csv", trainBaseNew)		
	csv_io.write_delimited_file("PreProcessData/test_PreProcess3.csv", testNew)
	print "Done."		
	
	
	
def PreProcess3c():


	trainBase = csv_io.read_data("PreProcessData/training_PreProcess3_TEMPa.csv", False)
	test = csv_io.read_data("PreProcessData/test_PreProcess3_TEMPa.csv", False)

	target = [x[0] for x in trainBase]
	train = [x[1:] for x in trainBase]
				
	
	DataClassList = csv_io.read_data("PreProcessData/DataClassList.csv", False)
	

	MTrain = []
	FTrain = []
	MTarget = []
	FTarget = []
		
			
	for index, data in enumerate(train):
		if ( data[0] == "0" ):
			MTrain.append([data[1]])
			MTarget.append(target[index])
			#print "M", data[1]
		if ( data[0] == "1" ):
			FTrain.append([data[1]])	
			FTarget.append(target[index])
			#print "F", data[1]
				
		

	Mneigh = LogisticRegression() 
	Fneigh = LogisticRegression()		
		
		
	Mneigh.fit(MTrain, MTarget) 
	Fneigh.fit(FTrain, FTarget) 
		


	for index, data in enumerate(train):
		if ( data[0] == "0" ):
			pred = Mneigh.predict_proba([data[1]])
			#print "M", data[1], pred[0][1], target[index]
			trainBase[index].append(pred[0][1])
		if ( data[0] == "1" ):
			pred = Fneigh.predict_proba([data[1]])
			#print "F", data[1], pred[0][1], target[index]
			trainBase[index].append(pred[0][1])


		
	for index, data in enumerate(test):
		if ( data[0] == "0" ):
			pred = Mneigh.predict_proba([data[1]])
			#print "M", data[1], pred[0][1], target[index]
			test[index].append(pred[0][1])
		if ( data[0] == "1" ):
			pred = Fneigh.predict_proba([data[1]])
			#print "F", data[1], pred[0][1], target[index]
			test[index].append(pred[0][1])	
		
	
	with open("PreProcessData/DataClassListLG.csv", "a") as myfile:
		myfile.write("NEW_Gender-Age\n")

	print "Writing Data"
	csv_io.write_delimited_file("PreProcessData/training_PreProcess3.csv", trainBase)		
	csv_io.write_delimited_file("PreProcessData/test_PreProcess3.csv", test)
	print "Done."	

	
	
		
if __name__=="__main__":

	print
	print "WARNING: This program appends features to the existing files. Do no run this more than once, otherwise redundant features will be appended to the list."
	print
	
	# need to add new line at end of file before appending data.
	#with open("PreProcessData/DataClassList.csv", "a") as myfile:
	#	myfile.write("\n")

	print "Creating Gender-Age-XFeature based class probabilities."
	PreProcess3()
	PreProcess3a()
	#PreProcess3b()
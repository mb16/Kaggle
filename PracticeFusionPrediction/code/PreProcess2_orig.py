#!/usr/bin/env python

import csv_io
import math

from math import log
import string


def toFloat(str):
	return float(str)


def PreProcess2():

	TrainData = csv_io.read_data("PreProcessData/training_PreProcess.csv", False)
	TestData = csv_io.read_data("PreProcessData/test_PreProcess.csv", False)

	# Replace M/F
	for TrainDatum in TrainData:
		if ( TrainDatum[1] == "M" ):
			TrainDatum[1] = 0;
		if ( TrainDatum[1] == "F" ):
			TrainDatum[1] = 1;	

	for TestDatum in TestData:
		if ( TestDatum[0] == "M" ):
			TestDatum[0] = 0;
		if ( TestDatum[0] == "F" ):
			TestDatum[0] = 1;	
	
	# Normalize Fields	
	NumFeatures = len(TestData[0])
	MinArray = [1000000.0]*NumFeatures
	MaxArray = [0.0]*NumFeatures

	TrainDataNew = []
	TestDataNew = []
	
	for TrainDatum in TrainData:	

		# for anomolously high heights, insert average height value, recompute bmi
		if ( toFloat(TrainDatum[3]) > 80):		
			#print "Problem: ",TrainDatum	
			TrainDatum[3] = 66
			TrainDatum[5] = 703 * toFloat(TrainDatum[4]) / (toFloat(TrainDatum[3]) * toFloat(TrainDatum[3]))
		
		# nothing can be down with these low heights, just set average height and cal bmi.
		if ( toFloat(TrainDatum[3]) < 40 and toFloat(TrainDatum[3]) > 10):		
			#print "Problem Low Height: ",TrainDatum	
			TrainDatum[3] = 66
			TrainDatum[5] = 703 * toFloat(TrainDatum[4]) / (toFloat(TrainDatum[3]) * toFloat(TrainDatum[3]))
			
		# note, heights below 10 appear to be typos, ie 5.7 is probably 5 foot 7 inches
		if ( toFloat(TrainDatum[3]) < 10):		
			#print "Problem Low Height: ",TrainDatum	, TrainDatum[3].split('.')[0], TrainDatum[3].split('.')[1]
			if ( len(TrainDatum[3].split('.')) == 1):
				TrainDatum[3] = toFloat(TrainDatum[3].split('.')[0]) * 12.0
			else:
				TrainDatum[3] = toFloat(TrainDatum[3].split('.')[0]) * 12.0 + toFloat(TrainDatum[3].split('.')[1])
			TrainDatum[5] = 703 * toFloat(TrainDatum[4]) / (toFloat(TrainDatum[3]) * toFloat(TrainDatum[3]))	
			#print TrainDatum[3]
			
		# check for too low weights	
		if ( toFloat(TrainDatum[4]) < 70):		
			print "Problem Low Weight: ",TrainDatum

		# check for too high weights, most seem to be decimal place problem, then recalc bmi	
		if ( toFloat(TrainDatum[4]) > 600):		
			#print "Problem High Weight: ",TrainDatum
			TrainDatum[4] = toFloat(TrainDatum[4]) / 10.0 # two entries probably have decimal shifted over one place.
			TrainDatum[5] = 703 * toFloat(TrainDatum[4]) / (toFloat(TrainDatum[3]) * toFloat(TrainDatum[3]))
			
		# check for high BP	
		if ( toFloat(TrainDatum[6]) > 250 or toFloat(TrainDatum[7]) > 200):		
			print "Problem High BP: ", TrainDatum		

		# check for low BP	
		if ( toFloat(TrainDatum[6]) < 40 or toFloat(TrainDatum[7]) < 40):		
			print "Problem Low BP: ", TrainDatum	
			
		# check for low bmi	
		if ( toFloat(TrainDatum[5]) < 10 ):		
			print "Problem Low bmi: ", TrainDatum				

		# check for high bmi	
		if ( toFloat(TrainDatum[5]) > 100 ):		
			print "Problem High bmi: ", TrainDatum, 703 * toFloat(TrainDatum[4]) / (toFloat(TrainDatum[3]) * toFloat(TrainDatum[3]))	
			
			
		# use feature + 1 to skip DM indicator value in first column
		for Feature in range(NumFeatures):
			if ( toFloat(TrainDatum[Feature + 1]) < MinArray[Feature] ) :
				MinArray[Feature] = toFloat(TrainDatum[Feature + 1])
			if ( toFloat(TrainDatum[Feature + 1]) > MaxArray[Feature] ) :
				MaxArray[Feature] = toFloat(TrainDatum[Feature + 1]	)	
		
		TrainDataNew.append(TrainDatum)
				
				
	for TestDatum in TestData:	
		
		offset = -1
		# for anomolously high heights, insert average height value, recompute bmi
		if ( toFloat(TestDatum[3 + offset]) > 80):		
			#print "Problem(test): ",TestDatum	
			TestDatum[3 + offset] = 66
			TestDatum[5 + offset] = 703 * toFloat(TestDatum[4 + offset]) / (toFloat(TestDatum[3 + offset]) * toFloat(TestDatum[3 + offset]))
		
		# nothing can be down with these low heights, just set average height and cal bmi.
		if ( toFloat(TestDatum[3 + offset]) < 40 and toFloat(TestDatum[3 + offset]) > 10):		
			#print "Problem(test) Low Height: ",TestDatum	
			TestDatum[3 + offset] = 66
			TestDatum[5 + offset] = 703 * toFloat(TestDatum[4 + offset]) / (toFloat(TestDatum[3 + offset]) * toFloat(TestDatum[3 + offset]))
			
		# note, heights below 10 appear to be typos, ie 5.7 is probably 5 foot 7 inches
		if ( toFloat(TestDatum[3 + offset]) < 10):	
			#print "Problem(test) Low Height: ", TestDatum		
			if ( len(TestDatum[3 + offset].split('.')) == 1):
				TestDatum[3 + offset] = toFloat(TestDatum[3 + offset].split('.')[0]) * 12.0
			else:
				TestDatum[3 + offset] = toFloat(TestDatum[3 + offset].split('.')[0]) * 12.0 + toFloat(TestDatum[3 + offset].split('.')[1])
			TestDatum[5 + offset] = 703 * toFloat(TestDatum[4 + offset]) / (toFloat(TestDatum[3 + offset]) * toFloat(TestDatum[3 + offset]))	
			#print TestDatum[3 + offset]
			
		# check for too low weights	
		if ( toFloat(TestDatum[4 + offset]) < 70):		
			print "Problem(test) Low Weight: ",TestDatum

		# check for too high weights, most seem to be decimal place problem, then recalc bmi	
		if ( toFloat(TestDatum[4 + offset]) > 600):		
			#print "Problem(test) High Weight: ",TestDatum
			TestDatum[4 + offset] = toFloat(TestDatum[4 + offset]) / 10.0 # two entries probably have decimal shifted over one place.
			TestDatum[5 + offset] = 703 * toFloat(TestDatum[4 + offset]) / (toFloat(TestDatum[3 + offset]) * toFloat(TestDatum[3 + offset]))
			
		# check for high BP	
		if ( toFloat(TestDatum[6 + offset]) > 250 or toFloat(TestDatum[7 + offset]) > 200):		
			print "Problem(test) High BP: ", TestDatum		

		# check for low BP	
		if ( toFloat(TestDatum[6 + offset]) < 40 or toFloat(TestDatum[7 + offset]) < 40):		
			print "Problem(test) Low BP: ", TestDatum	
			
		# check for low bmi	
		if ( toFloat(TestDatum[5 + offset]) < 10 ):		
			print "Problem(test) Low bmi: ", TestDatum				

		# check for high bmi	
		if ( toFloat(TestDatum[5 + offset]) > 100 ):		
			print "Problem(test) High bmi: ", TestDatum, 703 * toFloat(TestDatum[4 + offset]) / (toFloat(TestDatum[3 + offset]) * toFloat(TestDatum[3 + offset]))	
			
			
	
		for Feature in range(NumFeatures):
			if ( toFloat(TestDatum[Feature]) < MinArray[Feature] ) :
				MinArray[Feature] = toFloat(TestDatum[Feature])
			if ( toFloat(TestDatum[Feature]) > MaxArray[Feature] ) :
				MaxArray[Feature] = toFloat(TestDatum[Feature]	)					
	
	for TrainDatum in TrainDataNew:	
		for Feature in range(NumFeatures):
			if (MaxArray[Feature] - MinArray[Feature]) > 0.0:
				TrainDatum[Feature + 1] = (toFloat(TrainDatum[Feature + 1]) - MinArray[Feature] )/(MaxArray[Feature] - MinArray[Feature])

	for TestDatum in TestData:	
		for Feature in range(NumFeatures):
			if (MaxArray[Feature] - MinArray[Feature]) > 0.0:
				TestDatum[Feature] = (toFloat(TestDatum[Feature]) - MinArray[Feature] )/(MaxArray[Feature] - MinArray[Feature])

			
	print "MinArray: " , MinArray
	print "MaxArray: " ,MaxArray
			
			
	csv_io.write_delimited_file("PreProcessData/training_PreProcess2.csv", TrainDataNew)		
	csv_io.write_delimited_file("PreProcessData/test_PreProcess2.csv", TestData)	
	
	#var = raw_input("Enter to terminate.")	
			
								
if __name__=="__main__":
	PreProcess2()
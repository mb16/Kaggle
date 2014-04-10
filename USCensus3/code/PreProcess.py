#!/opt/local/bin python2.7

import csv_io
import math
from math import log
import string
import copy
import operator
import os

def toFloat(str):
	return float(str)

def PreProcess():
	PreProcessRun("training")
	PreProcessRun("test")	
	
def PreProcessRun(dataSet):
	print
	print "DataSet: ", dataSet
	
	if ( os.path.exists("PreProcessData/" + dataSet + "_PreProcess.csv") ):
		os.remove("PreProcessData/" + dataSet + "_PreProcess.csv")
	
	
	DataClassList = []
	
	f1 = open("../" + dataSet + "/" + dataSet + "_filev1.csv", 'r')
	f2 = open("PreProcessData/" + dataSet + "_PreProcess_temp.csv", 'w')
	for line in f1:
		newLine = ""
		gotQuote = False
		for c in line:
			if ( c == "\"" and gotQuote == False ):
				gotQuote = True
			elif ( c == "\"" and gotQuote == True ):			
				gotQuote = False
				
			if ( gotQuote == True and  c == ","):	
				continue
			elif(gotQuote == True):
				newLine += c
			else:
				if ( c == ","):
					newLine += "\t"
				else:
					newLine += c
				
		
		f2.write(newLine)
	f1.close()
	f2.close()

	
	data = csv_io.read_data("PreProcessData/" + dataSet + "_PreProcess_temp.csv", split="\t" ,skipFirstLine = False)

	weights = []
	first = True
	if ( dataSet == "training"):
		for row in data:
			if ( first  == True ) :
				first = False
				continue
			weights.append([row[13]])
			#print row[13]
		csv_io.write_delimited_file("PreProcessData/Weights.csv", weights)
	

	
	data = csv_io.read_data("PreProcessData/training_PreProcess_temp.csv", split="\t" ,skipFirstLine = True)	
	meanSum = [0.0] * 200
	meanCount = [0] * 200
	for index, val in enumerate(meanSum):
		meanCount[index] = 0
		meanSum[index] = 0.0
	
	for row in data:
		for index, val in enumerate(row):
			if ( isinstance(val, float) and val != 0.0):
				meanCount[index] += 1
				meanSum[index] += val
			#else:
				#print "skip: ", val
	
	for index, val in enumerate(meanSum):
		if meanCount[index] > 0:
			meanSum[index] = meanSum[index]/float(meanCount[index])
	

	data = csv_io.read_data("PreProcessData/" + dataSet + "_PreProcess_temp.csv", split="\t" ,skipFirstLine = False)
	SkipArr = [0,2,4,171]
		
	for index, item in enumerate(data[0]):
		#print item
		if index in SkipArr:
			continue
		#if "MOE_" in item:
		#	print "MOE_", item
		#	SkipArr.append(index)
		#	continue
		if ( index == 170 ):
			#DataClassList.insert(0, item)
			continue
		else:
			DataClassList.append(item)
			continue
	print "Len: ", len(data[0])

	first = True
	for item in data:
		#print item
		if ( first == True ):
			first = False
			continue
	
		rowNew = []

		for index, val in enumerate(item):
			if index in SkipArr:
				continue
			# in training this is the target value(append to beginning ), and in test this is the weight (just skip it) 
			if ( index == 170):
				#print "prepend", val
				if dataSet == "training":
					rowNew.insert(0, val)
				continue
			
		
			if ( val == "" or val == "NA" or val == "0" or val == "0.0" or val == 0 or val == 0.0):
				rowNew.append(meanSum[index]) 
			elif isinstance(val, str):
				rowNew.append(toFloat(val.replace("$", "")))	
			else:
				rowNew.append(val)
		
		csv_io.write_delimited_file("PreProcessData/" + dataSet + "_PreProcess.csv", [copy.deepcopy(rowNew)], filemode="a", delimiter="\t")


	
	csv_io.write_delimited_file("PreProcessData/DataClassList.csv", DataClassList)

	
	print "Done."		
								
if __name__=="__main__":
	PreProcess()
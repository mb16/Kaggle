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

def PreProcess2():
	PreProcessRun("training")
	PreProcessRun("test")	
	
def PreProcessRun(dataSet):
	print
	print "DataSet: ", dataSet
	
	print "Loading Data"
	data = csv_io.read_data("PreProcessData/" + dataSet + "_PreProcess1.csv", split="\t" ,skipFirstLine = False)
	print dataSet, "Size: ", len(data[0])
	
	if ( os.path.exists("PreProcessData/" + dataSet + "_PreProcess2.csv") ):
		os.remove("PreProcessData/" + dataSet + "_PreProcess2.csv")
	
	SkipArr = [0,2,4,172]

	
	DataClassList = csv_io.read_data("PreProcessData/DataClassList1.csv", False)
	DataClassListNew = []
	
	firstTime = True
	for index, item in enumerate(data):
	
		rowNew = []
		#print item
		
		for index, val in enumerate(item):
			if dataSet == "training" and (index - 1) in SkipArr:
				continue
			elif dataSet == "test" and index in SkipArr:
				continue
			rowNew.append(val)
		
			#print val
			if dataSet == "test" and firstTime == True:
				print DataClassList[index]
				DataClassListNew.append(DataClassList[index])
				
		csv_io.write_delimited_file("PreProcessData/" + dataSet + "_PreProcess2.csv", [copy.deepcopy(rowNew)], filemode="a", delimiter="\t")

		firstTime = False

	if dataSet == "test":
		csv_io.write_delimited_file("PreProcessData/DataClassList2.csv", DataClassListNew)

	
	print "Done."		
								
if __name__=="__main__":
	PreProcess2()
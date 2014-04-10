#!/opt/local/bin python2.7

import csv_io
import math
from math import log
import string
import copy
import operator
import os
import shutil

def toFloat(str):
	return float(str)

def PreProcess1():
	PreProcessRun("training")
	PreProcessRun("test")	
	
def PreProcessRun(dataSet):
	print
	print "DataSet: ", dataSet
	


	print "Loading Data"
	data = csv_io.read_data("PreProcessData/" + dataSet + "_PreProcess.csv", split="\t" ,skipFirstLine = False)
	print dataSet, "Size: ", len(data[0])
	
	if ( dataSet == "training" ): # do only once.
		shutil.copy2("PreProcessData/DataClassList.csv", "PreProcessData/DataClassList1.csv")
	
	DataClassList = csv_io.read_data("PreProcessData/DataClassList1.csv", False)	
	
	offset = 0;
	offset2 = 0;
	if ( dataSet == "test" ):
		offset = 1
		offset2 = -1
		
	print DataClassList
		
	print "Appending New Data"
	firstTime = True
	for row in data:
					
		text = ""
		
		val = row[136 + offset2]/row[139 + offset2]
		row.append(val)
		if (firstTime and dataSet == "training" ): # do only once.
			text = DataClassList[135 + offset][0] + "_DIV_" + DataClassList[139 + offset][0]
			csv_io.write_delimited_file("PreProcessData/DataClassList1.csv", [text], filemode="a")
		if (firstTime):					
			print row[136 + offset2],row[139 + offset2], val, text
	
				
		firstTime = False
	
	
	csv_io.write_delimited_file("PreProcessData/" + dataSet + "_PreProcess1.csv", data, delimiter="\t")
	
		
	print "Done."		
								
if __name__=="__main__":
	PreProcess1()
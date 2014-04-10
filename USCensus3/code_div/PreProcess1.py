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
	
	data = csv_io.read_data("../Data.csv", split="," ,skipFirstLine = False)
	
	
	
	print "Loading Divisor data"
	Loop1 = []
	Loop2 = []
	Loop3 = []
	for item in data:
		#print item
		Loop1.append(item[1])
		Loop2.append(item[2])
		Loop3.append(item[3])
		#print item[1], item[2], item[3], item


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
		#print row
		for index, item in enumerate(row): 
			#print index, item
			if (index == 0 and dataSet == "training" ):
				continue
			
			text = ""
			
			if (index + offset) < len(Loop1) and Loop1[index + offset] != "":
				#print item , row[int(Loop1[index + offset]) + offset2], index, index + offset, Loop1[index + offset]
				if ( row[int(Loop1[index + offset]) + offset2] == 0 ): # prevent div by zero.
					val = 0
				else:
					val = item/row[int(Loop1[index + offset]) + offset2]
				row.append(val)
				if (firstTime and dataSet == "training" ): # do only once.
					text = DataClassList[index - 1 + offset][0] + "_DIV_" + DataClassList[int(Loop1[index + offset]) - 1 + offset][0]
					csv_io.write_delimited_file("PreProcessData/DataClassList1.csv", [text], filemode="a")
				#if (firstTime):
				#	print val, text

				
			if (index + offset) < len(Loop2) and Loop2[index + offset] != "":
				#print item , row[int(Loop2[index + offset]) + offset2], index, index + offset, Loop2[index + offset]
				val = item/row[int(Loop2[index + offset]) + offset2]
				row.append(val)
				if (firstTime and dataSet == "training" ): # do only once.
					text = DataClassList[index - 1 + offset][0] + "____DIV____" + DataClassList[int(Loop2[index + offset]) - 1 + offset][0]
					csv_io.write_delimited_file("PreProcessData/DataClassList1.csv", [text], filemode="a")
				#if (firstTime):								
				#	print val, text

				
			if (index + offset) < len(Loop3) and Loop3[index + offset] != "":
				#print item , row[int(Loop3[index + offset]) + offset2], index, index + offset, Loop3[index + offset]
				val = item/row[int(Loop3[index + offset]) + offset2]
				row.append(val)
				if (firstTime and dataSet == "training" ): # do only once.
					text = DataClassList[index - 1 + offset][0] + "________DIV________" + DataClassList[int(Loop3[index + offset]) - 1 + offset][0]
					csv_io.write_delimited_file("PreProcessData/DataClassList1.csv", [text], filemode="a")
				#if (firstTime):					
				#	print val, text
			
		
		val = row[139 + offset2]/row[142 + offset2]
		row.append(val)
		if (firstTime and dataSet == "training" ): # do only once.
			text = DataClassList[138 + offset][0] + "________DIV________" + DataClassList[142 + offset][0]
			csv_io.write_delimited_file("PreProcessData/DataClassList1.csv", [text], filemode="a")
		if (firstTime):					
			print row[139 + offset2],row[142 + offset2], val, text

		
				
		firstTime = False
	

	
	
	csv_io.write_delimited_file("PreProcessData/" + dataSet + "_PreProcess1.csv", data, delimiter="\t")
	
		
	print "Done."		
								
if __name__=="__main__":
	PreProcess1()
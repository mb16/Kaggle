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

def PreProcess2():
	PreProcessRun("training")
	PreProcessRun("test")	
	
def PreProcessRun(dataSet):
	print
	print "DataSet: ", dataSet
	
	data = csv_io.read_data("data/CenPop2010_Mean_BG.txt", split="," ,skipFirstLine = True)
	
	
	
	print "Loading CenPop2010 data"
	CenPop2010_Population = {}
	CenPop2010_Latitude = {}
	CenPop2010_Longitude = {}
	for item in data:
		id = str(int(item[0])) + str(int(item[1])).rjust(3, '0') + str(int(item[2])).rjust(6, '0') + str(int(item[3]))
		#print "ID: ", id
		if id not in CenPop2010_Population:
			CenPop2010_Population[id] = item[4];
		else:
			print "Dup Error: " , id
		if id not in CenPop2010_Latitude:
			CenPop2010_Latitude[id] = item[5];	
		else:
			print "Dup Error: " , id
		if id not in CenPop2010_Longitude:
			CenPop2010_Longitude[id] = item[6];	
		else:
			print "Dup Error: " , id

	

	print "Loading Data"
	data = csv_io.read_data("PreProcessData/" + dataSet + "_PreProcess.csv", split="\t" ,skipFirstLine = False)
	print dataSet, "Size: ", len(data[0])
	
	
	offset = 0;
	if ( dataSet == "test" ):
		offset = -1
		
	print "Appending New Data"
	for row in data:
		#print row

		id = str(int(row[1 + offset])) + str(int(row[2 + offset])).rjust(3, '0') + str(int(row[3 + offset])).rjust(6, '0') + str(int(row[4 + offset]))
		#print id
		
		# population is already in the file.
		# if id in CenPop2010_Population:
			# row.append(CenPop2010_Population[id]);
		# else:
			# print "Find Error: " , id
		if id in CenPop2010_Latitude:
			row.append(CenPop2010_Latitude[id]);	
		else:
			print "Find Error: " , id
		if id in CenPop2010_Longitude:
			row.append(CenPop2010_Longitude[id]);	
		else:
			print "Find Error: " , id
	
	csv_io.write_delimited_file("PreProcessData/" + dataSet + "_PreProcess2.csv", data, delimiter="\t")
	
	if ( dataSet == "test" ): # do only once.
		shutil.copy2("PreProcessData/DataClassList.csv", "PreProcessData/DataClassList2.csv")
		csv_io.write_delimited_file("PreProcessData/DataClassList2.csv", ["CenPop2010_Latitude"], filemode="a")
		csv_io.write_delimited_file("PreProcessData/DataClassList2.csv", ["CenPop2010_Longitude"], filemode="a")
		
	print "Done."		
								
if __name__=="__main__":
	PreProcess2()
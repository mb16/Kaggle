#!/usr/bin/env python

import util
from collections import defaultdict
import numpy as np
import pandas as pd
import csv_io
import math
	
def get_date_dataframe(date_column):
	return pd.DataFrame({
		"SaleYear": [d.year for d in date_column],
		"SaleMonth": [d.month for d in date_column],
		"SaleDay": [d.day for d in date_column]
		}, index=date_column.index)	
	
	
	
def preprocess():

	train, test = util.get_train_test_df()

	
	columns = set(train.columns)
	#columns.remove("SalesID")
	#columns.remove("SalePrice")
	#columns.remove("saledate")

	#train_fea = get_date_dataframe(train["saledate"])
	#test_fea = get_date_dataframe(test["saledate"])

	#parseColumns = ["UsageBand"]
	parseColumns = [ "UsageBand","fiBaseModel","fiModelSeries","fiModelDescriptor","ProductSize","ProductGroup","Drive_System","Enclosure","Forks","Pad_Type","Ride_Control","Stick","Transmission","Turbocharged","Blade_Extension","Blade_Width","Enclosure_Type","Engine_Horsepower","Hydraulics","Pushblock","Ripper","Scarifier","Tip_ControlCoupler","Coupler_System","Grouser_Tracks","Hydraulics_Flow","Track_Type","Thumb","Pattern_Changer","Grouser_Type","Backhoe_Mounting","Blade_Type","Travel_Controls","Differential_Type","Steering_Controls"]
	
	#"auctioneerID","state","ProductGroupDesc",,"fiSecondaryDesc"
	# this is redundant "fiModelDesc", and has too many options...
	
	# Q, AC, AL AR AS
	
	colDict = {}
	for col in parseColumns:
		colDict[col] = []
		
	colMap = {}	
	notInTest = []
	for index, col in enumerate(train.columns):
		print "MAP:", col, index
		colMap[col] = index
		if col in parseColumns:
			#print "start"			
			s = set(x for x in train[col].fillna(0)) # 0 if x == "" or not isinstance(x, float) else x
			s.update(x for x in test[col].fillna(0)) # math.isnan(x)
			
			colDict[col] = s
			print s
			
			if col == "fiBaseModel":
				a = set(x for x in train[col].fillna(0))
				b = set(x for x in test[col].fillna(0))		
				print "fiBaseModel"
				print
				print
				# found 11 type in test not in train
				print [x for x in b if x not in a]
				print
				print
				# found several hundred in train that are not in test, try dropping these...
				print [x for x in a if x not in b]
				notInTest = [x for x in a if x not in b]

				
	SaleIDArr = []		
	trainSalePriceArr = []

	count = 0
	csv_io.delete_file("train1.csv")
	for row in train.iterrows():
		trainSalePrice = []
	
		rowVals = row[1].fillna(0)
		newSet = []
		newRow = []
		
		if rowVals["fiBaseModel"] not in notInTest:
			continue
		
		trainSalePrice.append(rowVals["SalePrice"])
		trainSalePriceArr.append(trainSalePrice)
		
		SaleID = []
		SaleID.append(rowVals["SalesID"])
		SaleIDArr.append(SaleID)
		
		for col in colDict.keys():
			for val in colDict[col]:
				if val == rowVals[col] :
					newRow.append(1)
				else:
					newRow.append(0)

		#newRow.append(rowVals["YearMade"]) # need to calculate age, sale date minus year
		newRow.append(rowVals["MachineHoursCurrentMeter"])
		
		count += 1
		if count % 10000 == 0:
			print "Count", count
			
		newSet.append(newRow)
		csv_io.write_delimited_file("train1.csv", newSet ,header=None, delimiter=",", filemode="a")

		
	csv_io.write_delimited_file("target.csv", trainSalePriceArr ,header=None, delimiter=",")
	csv_io.write_delimited_file("train_salesID.csv", SaleIDArr ,header=None, delimiter=",")		
	# -------------------------------------------	
	
	SaleIDArr = []
	
	count = 0
	csv_io.delete_file("test1.csv")
	for row in test.iterrows():

		rowVals = row[1].fillna(0)
		newSet = []
		newRow = []
		
		SaleID = []
		SaleID.append(rowVals["SalesID"])
		SaleIDArr.append(SaleID)
		
		for col in colDict.keys():
			for val in colDict[col]:
				if val == rowVals[col] :
					newRow.append(1)
				else:
					newRow.append(0)

		#newRow.append(rowVals["YearMade"]) # need to calculate age, sale date minus year
		newRow.append(rowVals["MachineHoursCurrentMeter"])
		
		count += 1
		if count % 10000 == 0:
			print "Count", count
			
		newSet.append(newRow)
		csv_io.write_delimited_file("test1.csv", newSet ,header=None, delimiter=",", filemode="a")
	
	csv_io.write_delimited_file("test_salesID.csv", SaleIDArr ,header=None, delimiter=",")		
	


if __name__=="__main__":
	preprocess()






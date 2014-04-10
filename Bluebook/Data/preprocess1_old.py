#!/usr/bin/env python

import util
from collections import defaultdict
import numpy as np
import pandas as pd

	
def get_date_dataframe(date_column):
	return pd.DataFrame({
		"SaleYear": [d.year for d in date_column],
		"SaleMonth": [d.month for d in date_column],
		"SaleDay": [d.day for d in date_column]
		}, index=date_column.index)	
	
	
	
def preprocess1():

	train, test = util.get_train_test_df()

	columns = set(train.columns)
	columns.remove("SalesID")
	columns.remove("SalePrice")
	columns.remove("saledate")

	train_fea = get_date_dataframe(train["saledate"])
	test_fea = get_date_dataframe(test["saledate"])

	for col in columns:
		print col
		types = set(type(x) for x in train[col])
		if str in types:
			print "in:", col
			s = set(x for x in train[col])
			str_to_categorical = defaultdict(lambda: -1, [(x[1], x[0]) for x in enumerate(s)])
			
			#print str_to_categorical
			#return
			
			train_fea = train_fea.join(pd.DataFrame({col: [str_to_categorical[x] for x in train[col]]}, index=train.index))
			test_fea = test_fea.join(pd.DataFrame({col: [str_to_categorical[x] for x in test[col]]}, index=test.index))
		else:
			train_fea = train_fea.join(train[col])
			test_fea = test_fea.join(test[col])

			
	train_fea.to_csv("train_fea.csv", index=False)				
	test_fea.to_csv("test_fea.csv", index=False)		
			
			
	train["SalePrice"].to_csv("target.csv", index=False)
	
	train["SalesID"].to_csv("train_salesID.csv", index=False)			
	test["SalesID"].to_csv("test_salesID.csv", index=False)

if __name__=="__main__":
	preprocess1()






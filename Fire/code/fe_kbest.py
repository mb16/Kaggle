#!/usr/bin/env python

from sklearn import svm
import csv_io
import score
import math
from math import log
from sklearn import cross_validation

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
import scipy

import gc


    
def run_feature_select(SEED):

    
    numFeatures = 40
    
    trainBaseTarget = pd.read_csv('../preprocessdata/pre_shuffled_target.csv')
    trainBase = pd.read_csv('../preprocessdata/pre_departition_train.csv')

    

    m = SelectKBest(f_regression, k=numFeatures)
    m.fit(trainBase, trainBaseTarget)
    cols = m.get_support(indices=False)
  
    print(m.scores_) 
     
     
    p = np.vstack([trainBase.columns,m.scores_])
    submission = pd.DataFrame(p.T, columns = None)
    submission.to_csv("../featureanalysis/SelectKBest_" + str(numFeatures) + ".csv")   
        
        
        
    gc.collect()      
    for index, col in enumerate(trainBase.columns):
        print("Column: " + col)
        if cols[index] == False and col != "var11" and col != "id":
            print("Dropping")
            trainBase.drop([col], axis=1, inplace=True)
    gc.collect()
    trainBase.to_csv("../models/SelectKBest_" + str(numFeatures) +  "_train.csv", index = False)
    
    
    gc.collect()

    test = pd.read_csv('../preprocessdata/pre_departition_test.csv')

    for index, col in enumerate(test.columns):
        print("Column: " + col)
        if cols[index] == False and col != "var11" and col != "id":
            print("Dropping")
            test.drop([col], axis=1, inplace=True)
    gc.collect()
    test.to_csv("../models/SelectKBest_" + str(numFeatures) + "_test.csv", index = False)  
    gc.collect()                
              

if __name__=="__main__":
    run_feature_select(448)
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
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import scipy

import gc

import datetime
import random

from sklearn import preprocessing
from sklearn.utils import shuffle    

import score    
    
def run_feature_select(SEED):

    # inlcude pos and neg ~100
    #sets = {
    #    #'single_feaure_Rando_abs': 0.061, # does nat capture neg data. 
    #    'single_feaure_Gradi_abs': 0.05, 
    #    'single_feaure_Ridge_abs': 0.03, 
    #    }

    # inlcude pos and neg ~30
    sets = {
        'single_feaure_Rando_abs': 0.04, 
        'single_feaure_Gradi_abs': 0.04, 
        'single_feaure_Ridge_abs': 0.02, 
        }

    # large sets
    #sets = {
    #    'single_feaure_Rando': 0.061, 
    #    'single_feaure_Gradi': 0.05, 
    #    'single_feaure_Ridge': 0.032, 
    #    }

    # Small sets.
    #sets = {
    #    'single_feaure_Rando': 0.065, 
    #    'single_feaure_Gradi': 0.075, 
    #    'single_feaure_Ridge': 0.045, 
    #    }
    
    
    trainBase = pd.read_csv('../data/pre_shuffled_train.csv')
    test = pd.read_csv('../data/pre_shuffled_test.csv')
    

    for featureSet, threshold in sets.items():

        print(featureSet + " " + str(threshold))

        columnScores = pd.read_csv('../featureanalysis/' + featureSet.replace("_abs", "") + '.csv', index_col=None, header=None)
    
       
        scores = {}
        for index, row in columnScores.iterrows():
            k, v = row
            scores[k] = v
            

            
        gc.collect()      
        submission = pd.DataFrame(trainBase, columns = test.columns)
        for col in submission.columns:
            print("Column: " + col + " " + str(scores[col]))
            if abs(scores[col]) < threshold and col != "id" and col != "var11":
                print("Dropping")
                submission.drop([col], axis=1, inplace=True)
        gc.collect()
        submission.to_csv("../models/" + featureSet + "_" + str(threshold) +  "_train.csv", index = False)
        gc.collect()
    
    
        submission = pd.DataFrame(test, columns = test.columns)
        for col in submission.columns:
            print("Column: " + col + " " + str(scores[col]))
            if abs(scores[col]) < threshold and col != "id" and col != "var11":
                print("Dropping")
                submission.drop([col], axis=1, inplace=True)
        gc.collect()
        submission.to_csv("../models/" + featureSet + "_" + str(threshold) + "_test.csv", index = False)  
        gc.collect()                
              

if __name__=="__main__":
    run_feature_select(448)
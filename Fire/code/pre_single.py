#!/usr/bin/env python



import csv_io
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import numpy as np
import pandas as pd


def toFloat(str):
    return float(str)


def run_rf(SEED):

    target = pd.read_csv('../data/pre_shuffled_target.csv')
    target = np.ravel(target.values)

    weights = pd.read_csv('../data/weights.csv')
    weights = np.ravel(weights.values)

    trainBase = pd.read_csv('../data/pre_shuffled_train.csv')
    test = pd.read_csv('../data/pre_shuffled_test.csv') 




    NumFeatures = 30
    clf = RandomForestRegressor(n_estimators=30, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, min_density=None, compute_importances=True) 
    #clf = ExtraTreesRegressor(n_estimators=150, min_density=0.02, n_jobs=1, criterion='gini', bootstrap=True,compute_importances=True)
    print("Training")
    clf.fit(trainBase.values, target, sample_weight = weights)
        
        
    print("Computing Importances")
    importances = clf.feature_importances_
    print(importances)
    


    importancesSorted = sorted(importances, reverse=True)
    print (str(len(importancesSorted)) + " importances")
    

    threshold = 1.0
    if ( len(importancesSorted) > NumFeatures):
        threshold = importancesSorted[NumFeatures]
    
    print("Threshold: " + str(threshold))


    

    DataClassListNew = []
    print(trainBase.columns.values)
    for DataIndex, DataClass in enumerate(trainBase.columns.values):
        print(str(DataIndex) + " " + DataClass + ", " +  str(importances[DataIndex]))
        DataClassListNew.append([DataClass, importances[DataIndex]])
        
        if ( importances[DataIndex] < threshold and DataClass != "id" and DataClass != "var11" ):  # don't drop id or weights column.    
            trainBase.drop([DataClass], axis=1, inplace=True)
            test.drop([DataClass], axis=1, inplace=True)
               
        
        
    csv_io.write_delimited_file("../preprocessdata/DataClassList_Importances_RF.csv", DataClassListNew)

    
    submission = pd.DataFrame(trainBase) 
    submission.to_csv("../data/pre_rf_train.csv", index = False)


    submission = pd.DataFrame(test) 
    submission.to_csv("../data/pre_rf_test.csv", index = False) 

   
if __name__=="__main__":
    run_rf(448)
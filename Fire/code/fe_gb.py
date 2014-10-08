#!/usr/bin/env python



import csv_io
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd


def toFloat(str):
    return float(str)


def run_rf(SEED):

    target = pd.read_csv('../preprocessdata/pre_shuffled_target_class.csv')
    target = np.ravel(target.values)

    #weights = pd.read_csv('../data/weights.csv')
    #weights = np.ravel(weights.values)

    trainBase = pd.read_csv('../preprocessdata/pre_departition_train.csv')





    NumFeatures = 30
    
    
    clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

    print("Training")
    clf.fit(trainBase.values, target) # sample_weight = weights)
        
        
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
            #test.drop([DataClass], axis=1, inplace=True)
               
        
        
    csv_io.write_delimited_file("../featureanalysis/DataClassList_Importances_GB.csv", DataClassListNew)

    
    submission = pd.DataFrame(trainBase) 
    submission.to_csv("../preprocessdata/pre_gb_train.csv", index = False)




    # load and run after train because data is too large for memeory.
    test = pd.read_csv('../preprocessdata/pre_departition_test.csv') 

    for DataIndex, DataClass in enumerate(test.columns.values):
        print(str(DataIndex) + " " + DataClass + ", " +  str(importances[DataIndex]))
        
        if ( importances[DataIndex] < threshold and DataClass != "id" and DataClass != "var11" ):  # don't drop id or weights column.    
            #trainBase.drop([DataClass], axis=1, inplace=True)
            test.drop([DataClass], axis=1, inplace=True)

    submission = pd.DataFrame(test) 
    submission.to_csv("../preprocessdata/pre_gb_test.csv", index = False) 

   
if __name__=="__main__":
    run_rf(448)
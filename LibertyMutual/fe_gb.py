#!/usr/bin/env python



import csv_io
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd


def toFloat(str):
    return float(str)


def run_rf(SEED, col):

    dset = "3"  

    target = pd.read_csv('../preprocess/pre_shuffled_target_' + col + '.csv')
    target = np.ravel(target.values)


    trainBase = pd.read_csv('../preprocess/pre_shuffled_train' + dset + '.csv')
    trainBase.drop(['PIDN'], axis=1, inplace=True)




    NumFeatures = 500 # should go to atleast 500, but 0's dont start til about 2500
    
    
    clf = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=3000, subsample=1.0, min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False)

    print("Training")
    clf.fit(trainBase.values, target) # sample_weight = weights)
        
        
    print("Computing Importances")
    importances = clf.feature_importances_
    print(importances)
    


    importancesSorted = sorted(importances, reverse=True)
    print (str(len(importancesSorted)) + " importances")
    

    threshold = 0.0
    if ( len(importancesSorted) > NumFeatures):
        threshold = importancesSorted[NumFeatures]
    
    print("Threshold: " + str(threshold))


    
    
    trainBase = pd.read_csv('../preprocess/pre_shuffled_train' + dset + '.csv')

    DataClassListNew = []
    print(trainBase.columns.values)
    for DataIndex, DataClass in enumerate(trainBase.columns.values):
        
        if (DataClass != "PIDN"):
            print(str(DataIndex) + " " + DataClass + ", " +  str(importances[DataIndex - 1])) # -1 because we droppped id above
            DataClassListNew.append([DataClass, importances[DataIndex - 1]])
        
        if ( importances[DataIndex - 1] <= threshold and DataClass != "PIDN"):  # don't drop id and -1 because we droppped id above
            trainBase.drop([DataClass], axis=1, inplace=True)
            #test.drop([DataClass], axis=1, inplace=True)
               
        
        
    csv_io.write_delimited_file("../featureanalysis/DataClassList" + dset + "_Importances_GB_" + col + ".csv", DataClassListNew)

    
    submission = pd.DataFrame(trainBase) 
    submission.to_csv("../models/gb" + dset + "_train_" + col + ".csv", index = False)




    # load and run after train because data is too large for memeory.
    test = pd.read_csv('../preprocess/pre_shuffled_test' + dset + '.csv') 

    for DataIndex, DataClass in enumerate(test.columns.values):
        
        if (DataClass != "PIDN"):
            print(str(DataIndex) + " " + DataClass + ", " +  str(importances[DataIndex - 1])) # -1 because we droppped id above
        
        if ( importances[DataIndex - 1] <= threshold and DataClass != "PIDN"):  # don't drop id or weights column.  and -1 because we droppped id above
            #trainBase.drop([DataClass], axis=1, inplace=True)
            test.drop([DataClass], axis=1, inplace=True)

    submission = pd.DataFrame(test) 
    submission.to_csv("../models/gb" + dset + "_test_" + col + ".csv", index = False) 

   
if __name__=="__main__":
    
    colsTarget = ['Ca','P','pH','SOC','Sand']     

    for Index, col in enumerate(colsTarget):
        run_rf(448, col)
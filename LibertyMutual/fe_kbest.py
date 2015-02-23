#!/usr/bin/env python


import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression


import gc


    
def run_feature_select(SEED, colTarget):

    dset = "5"    
    
    numFeatures = 1000
    
    trainBaseTarget = pd.read_csv('../preprocess/pre_shuffled_target_' + colTarget + '.csv')
    trainBase = pd.read_csv('../preprocess/pre_shuffled_train' + dset + '.csv')


    
    trainBase.drop(['PIDN'], axis=1, inplace=True)

    #m = SelectKBest(f_regression, k=numFeatures)
    m = SelectKBest(f_regression, k='all')
    m.fit(trainBase, trainBaseTarget)
    cols = m.get_support(indices=False)
  
    print(m.scores_) 
     
     
    p = np.vstack([trainBase.columns,m.scores_])
    submission = pd.DataFrame(p.T, columns = None)
    submission.to_csv("../featureanalysis/SelectKBest" + dset + "_" + str(numFeatures) + "_" + colTarget +".csv")   
        
        
    # reread since we dropped a non-float column and the dataset is small....    
    trainBase = pd.read_csv('../preprocess/pre_shuffled_train' + dset + '.csv')    
        
    gc.collect()      
    for index, col in enumerate(trainBase.columns):
        print("Column: " + col)
        if col != "PIDN" and cols[index - 1] == False: # -1 since we dropped a column above.
            print("Dropping")
            trainBase.drop([col], axis=1, inplace=True)
    gc.collect()
    trainBase.to_csv("../models/SelectKBest" + dset + "_" + str(numFeatures) + "_train_" + colTarget + ".csv", index = False)
    
    
    gc.collect()

    test = pd.read_csv('../preprocess/pre_shuffled_test' + dset + '.csv')

    for index, col in enumerate(test.columns):
        print("Column: " + col)
        if  col != "PIDN" and cols[index - 1] == False: # -1 since we dropped a column above.
            print("Dropping")
            test.drop([col], axis=1, inplace=True)
    gc.collect()
    test.to_csv("../models/SelectKBest" + dset + "_" + str(numFeatures) + "_test_" + colTarget + ".csv", index = False)  
    gc.collect()                
              

if __name__=="__main__":
    
    colsTarget = ['Ca','P','pH','SOC','Sand']     

    for Index, col in enumerate(colsTarget):
        run_feature_select(448, col)
#!/usr/bin/env python


import csv_io
import score

from sklearn import cross_validation
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


import gc

import datetime

 
import os

    
def run_stack(SEED):

    model = "Lasso"
    lossThreshold = 0.3 

    trainBaseTarget = pd.read_csv('../preprocessdata/pre_shuffled_target.csv')
    trainBaseOrig = pd.read_csv('../models/' + model + '_train.csv')
    trainBaseWeight = trainBaseOrig['var11']
    testOrig = pd.read_csv('../models/' + model + '_test.csv')


    targetBase = np.nan_to_num(np.array(trainBaseTarget))


    trainBaseID = trainBaseOrig['id']
    testID = testOrig['id']    

    
    avg = 0
    NumFolds = 5

    avgLast = avg
    bestAvg = avg
    bestAlpha = 0


    stackFiles = []
    for filename in os.listdir("../predictions"):
        parts = filename.split("_")
        if ( filename[0:5] == "Stack" and float(parts[2]) > lossThreshold):

            stackFiles.append(filename)
    
    
    trainBase = np.zeros((len(trainBaseOrig), len(stackFiles)))
    test = np.zeros((len(testOrig), len(stackFiles)))
    
    print("Loading Data")
    for fileNum, file in enumerate(stackFiles):
        print(file)
        trn = csv_io.read_data("../predictions/Target_" + file, split="," ,skipFirstLine = True) # skip first because of header.
        for row, datum in enumerate(trn):
            trainBase[row, fileNum] = datum[1] # -1 because we skil 
        
        tst = csv_io.read_data("../predictions/" + file, split="," ,skipFirstLine = True) # skip first because of header.
        for row, datum in enumerate(tst):
            test[row, fileNum] = datum[1]

    np.savetxt('temp/dataset_blend_train.txt', trainBase)
    np.savetxt('temp/dataset_blend_test.txt', test)
    print("Num file processed: " + " " + str(len(stackFiles))  + " " +  "Threshold: " + str(lossThreshold))

    
    print("Starting Scale")

    allVals = np.vstack((trainBase,test))    
    
    scl = StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(allVals) # should fit on the combined sets.
        
    trainBase= scl.transform(trainBase)
    test = scl.transform(test)
      
 
    
    
    print("Starting Blend")


    
    
    print("Begin Training")
    
    lenTrainBase = len(trainBase)
    lenTest = len(test)
    
    

    gc.collect()
    
    for a in np.logspace(-6, -.5, 10): # best values seem to be slightly greater than 0.
              
        
        clf = Lasso(alpha=a)
        print(clf)
        avg = 0
    

            
        coef_dataset = np.zeros((len(stackFiles),NumFolds))

        
        foldCount = 0

        Folds = cross_validation.KFold(lenTrainBase, n_folds=NumFolds, indices=True)
            
        for train_index, test_index in Folds:
    
            print()
            print ("Iteration: " + str(foldCount))
            
            
            now = datetime.datetime.now()
            print(now.strftime("%Y/%m/%d %H:%M:%S"))    
    
    
            target = [targetBase[i] for i in train_index]
            train = [trainBase[i] for i in train_index]
            weight = [trainBaseWeight[i] for i in train_index]
            
            targetTest = [targetBase[i] for i in test_index]    
            trainTest = [trainBase[i] for i in test_index]    
            weightTest = [trainBaseWeight[i] for i in test_index]
            

            #print "LEN: ", len(train), len(target)
            
            
            target = np.array(np.reshape(target, (-1, 1)) )           
            #train = np.array(np.reshape(train, (-1, 1))  ) 
            weight = np.array(np.reshape(weight, (-1, 1)))              
    
            targetTest = np.array(np.reshape(targetTest, (-1, 1)) )  
            #trainTest = np.array(np.reshape(trainTest, (-1, 1)) )  
            weightTest = np.array(np.reshape(weightTest, (-1, 1)))              
            

            clf.fit(train, target)
            predicted = clf.predict(trainTest) 

        
            print(str(score.normalized_weighted_gini(targetTest.ravel(), predicted.ravel(), weightTest.ravel())))
            avg += score.normalized_weighted_gini(targetTest.ravel(), predicted.ravel(), weightTest.ravel())/NumFolds
   
                 
            coef_dataset[:, foldCount] = clf.coef_
        
                
            foldCount = foldCount + 1
        
            #break
        
        
        coefs = coef_dataset.mean(1)
        print(coefs)        
        sorted_coefs = sorted(coefs)
        print("len coefs: " + str(len(sorted_coefs)))
   
        coefsAboveZero = [i for i in coefs if i > 0.0]   
        print(str(len(coefsAboveZero)))
   
        print ("------------------------Average: " + str(avg))               
  
        if avg > bestAvg:
            bestAvg = avg
            bestAlpha = a
  
  
    print("bestAvg: " + str(bestAvg))
    print("bestAlpha: " + str(bestAlpha))
  
    
    
if __name__=="__main__":
    run_stack(448)
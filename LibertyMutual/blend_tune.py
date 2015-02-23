#!/usr/bin/env python

import csv_io


from sklearn import cross_validation
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


import gc

import datetime

 
import os
from sklearn.metrics import mean_squared_error
import math  
        

dset = "4"      
    
def run_stack(SEED, col):


    model = "Lasso"
    lossThreshold = 0.48

    trainBaseTarget = pd.read_csv('../preprocess/pre_shuffled_target_' + col + '.csv')
    trainBaseOrig = pd.read_csv('../models/' + model + dset + '_train_' + col + '.csv')
    testOrig = pd.read_csv('../models/' + model + dset + '_test_' + col + '.csv')

    targetBase = np.nan_to_num(np.array(trainBaseTarget))

    #print(trainBase.columns)
    trainBaseID = trainBaseOrig['PIDN']
    testID = testOrig['PIDN']  
  
  
      
    avg = 0
    NumFolds = 10

	# ----------------------
	
    stackFiles = []
    for filename in os.listdir("../predictions"):
        parts = filename.split("_")
        if ( filename[0:5] == "Stack" and float(parts[2]) < lossThreshold): # and "Lasso" in filename 

            stackFiles.append(filename)
    
    
    trainBase = np.zeros((len(trainBaseOrig), len(stackFiles)))
    test = np.zeros((len(testOrig), len(stackFiles)))
    
	
	
    # first col is PIDN, after that we have 'Ca','P','pH','SOC','Sand', so we need to add 1
    if col == 'Ca':
        targetCol = 1
    elif col == 'P':
        targetCol = 2
    elif col == 'pH':
        targetCol = 3
    elif col == 'SOC':
        targetCol = 4
    elif col == 'Sand':
        targetCol = 5

	
	
	
	
    print("Loading Data")
    for fileNum, file in enumerate(stackFiles):
        print(file)
        trn = csv_io.read_data("../predictions/Target_" + file, split="," ,skipFirstLine = True) # skip first because of header.
        for row, datum in enumerate(trn):
            trainBase[row, fileNum] = datum[targetCol] 
        
        tst = csv_io.read_data("../predictions/" + file, split="," ,skipFirstLine = True) # skip first because of header.
        for row, datum in enumerate(tst):
            test[row, fileNum] = datum[targetCol]

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

		
	
	
	# --------------------------------

    
    
    
    print ("Data size: " + str(len(trainBase)) + " " + str(len(test)))
    #dataset_blend_train = np.zeros((len(trainBase), 1))
    #dataset_blend_test = np.zeros((len(test), 1))
    
    #averageSet = []

    
    print("Begin Training")
    
    lenTrainBase = len(trainBase)
    #lenTest = len(test)
    
    
    avg = 1.0
    #avgLast = avg
    bestAvg = avg
    bestAlpha = 0    
    


    gc.collect()
    
    for a in np.logspace(-8, -.5, 50): # best values seem to be slightly greater than 0.
               
        
        clf = Lasso(alpha=a)
        #print(clf)
        avg = 0
    

        coef_dataset = np.zeros((len(stackFiles),NumFolds))    
        #dataset_blend_test_set = np.zeros((lenTest, NumFolds))

        
        foldCount = 0

        Folds = cross_validation.KFold(lenTrainBase, n_folds=NumFolds, indices=True)
            
        for train_index, test_index in Folds:
    
            #print()
            #print ("Iteration: " + str(foldCount))
            
            
            now = datetime.datetime.now()
            #print(now.strftime("%Y/%m/%d %H:%M:%S"))    
    
    
            target = [targetBase[i] for i in train_index]
            train = [trainBase[i] for i in train_index]

            
            targetTest = [targetBase[i] for i in test_index]    
            trainTest = [trainBase[i] for i in test_index]    

            

            
            
            target = np.array(np.reshape(target, (-1, 1)) )           
           
    
            targetTest = np.array(np.reshape(targetTest, (-1, 1)) )  
          
            


            clf.fit(train, target)
            predicted = clf.predict(trainTest) 

            avg += math.sqrt(mean_squared_error(targetTest, predicted))/NumFolds
                 
            coef_dataset[:, foldCount] = clf.coef_   
                
            foldCount = foldCount + 1
        
            #break
        

        
        coefs = coef_dataset.mean(1)
        #print(coefs)        
        sorted_coefs = sorted(coefs)
        #print("len coefs: " + str(len(sorted_coefs)))
   
        coefsAboveZero = [i for i in coefs if i > 0.0]   
        #print(str(len(coefsAboveZero)))
   
        print ("------------------------Average: " + str(avg))               
  
        if avg < bestAvg:
            bestAvg = avg
            bestAlpha = a
  
  
    print("bestAvg: " + str(bestAvg))
    print("bestAlpha: " + str(bestAlpha))

                           
    
    
if __name__=="__main__":
    
    colsTarget = ['Ca','P','pH','SOC','Sand']     

     
    for Index, col in enumerate(colsTarget):    
        run_stack(448,col)
      
        
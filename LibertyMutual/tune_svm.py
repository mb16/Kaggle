#!/usr/bin/env python





from sklearn import cross_validation
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
import gc
import datetime
from sklearn.metrics import mean_squared_error
import math  
from sklearn.svm import SVC, SVR
def run_stack(SEED, col):

    dset = "2"  

    trainBaseTarget = pd.read_csv('../preprocess/pre_shuffled_target_' + col + '.csv')
    trainBase = pd.read_csv('../models/rf' + dset + '_train_' + col + '.csv')
    #trainBase = pd.read_csv('../preprocess/pre_shuffled_train' + dset + '.csv')
    trainBase.drop(['PIDN'], axis=1, inplace=True)


    #test = pd.read_csv('../data/pre_shuffled_test.csv')

    columns = trainBase.columns    
    columnsHighScore = trainBase.columns 


    print(trainBase.columns)
    
    trainBase = np.nan_to_num(np.array(trainBase))
    targetBase = np.nan_to_num(np.array(trainBaseTarget))
    #test = np.nan_to_num(np.array(test))
    
    gc.collect()   
   
    
    avg = 1.0
    avgLast = avg
    bestAvg = avg
    bestC = 0
    bestG = 0
    NumFolds = 5


   

    
    
    
    print ("Data size: " + str(len(trainBase)))
    print("Begin Training")
    
    lenTrainBase = len(trainBase)
   

    gc.collect()
    
    
    CC = [6,5,7,4,8,3,9,2,10,1]
    GG = [-6,-7,-5,-8,-4,-9,-3,-10,-2,-1]  
    
    for C in CC:
        for G in GG:
        
            print("g:" + str(G) +  "C:" + str(C))
            
            clf = SVR(C=10**C, kernel='rbf', degree=3, gamma=10**G, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False)
    
            #print(clf)
            avg = 0
        
            #coef_dataset = np.zeros((len(columns),NumFolds))
       
            foldCount = 0
    
            Folds = cross_validation.KFold(lenTrainBase, n_folds=NumFolds, indices=True)
                
            for train_index, test_index in Folds:
        
                #print()
                #print ("Iteration: " + str(foldCount))
                
                
                now = datetime.datetime.now()
                print(now.strftime("%Y/%m/%d %H:%M:%S"))    
        
        
                target = [targetBase[i] for i in train_index]
                train = [trainBase[i] for i in train_index]
    
                
                targetTest = [targetBase[i] for i in test_index]    
                trainTest = [trainBase[i] for i in test_index]    
    
                
    
                #print "LEN: ", len(train), len(target)
                
                
                target = np.array(np.reshape(target, (-1, 1)) ) 
                target = target.flatten()
                #train = np.array(np.reshape(train, (-1, 1))  ) 
                
        
                targetTest = np.array(np.reshape(targetTest, (-1, 1)) )  
                #trainTest = np.array(np.reshape(trainTest, (-1, 1)) )  
                 
                
    
                #clf.fit(train, target, sample_weight = weight
                clf.fit(train, target)
                predicted = clf.predict(trainTest) 
     
     
                #print(target.shape) 
                #print(predicted.shape)
      
                print(str(math.sqrt(mean_squared_error(targetTest, predicted))))
                avg += math.sqrt(mean_squared_error(targetTest, predicted))/NumFolds
    
                     
                #coef_dataset[:, foldCount] = clf.coef_                 
    
                foldCount = foldCount + 1
            
                #break
         
        
            #coefs = coef_dataset.mean(1)
            #print(coefs)        
            #sorted_coefs = sorted(coefs)
            #print("len coefs: " + str(len(sorted_coefs)))
       
            #coefsAboveZero = [i for i in coefs if i > 0.0]   
            #print(str(len(coefsAboveZero)))
       
            print ("------------------------Average: " + str(avg))               
      
            if avg < bestAvg:
                bestAvg = avg
                bestC = C
                bestG = G
      
  
    print("bestAvg: " + str(bestAvg))
    print("bestC: " + str(bestC))
    print("bestG: " + str(bestG))  
    
if __name__=="__main__":
    
    # Ca C=6 G=-4    
    # P C=1 G=-3      
    # pH C=2 G=-4      
    # SOC C=1 G=-4     
    # Sand C=6 G=-3    
    
    colsTarget = ['Ca','P','pH','SOC','Sand']     
    colsTarget = ['Sand'] 

    for Index, col in enumerate(colsTarget):    
        run_stack(448,col)
#!/usr/bin/env python



import score

from sklearn import cross_validation


from sklearn.linear_model import Lasso

import numpy as np
import pandas as pd


import gc

import datetime
  
    
import score
    
def run_stack(SEED):



    trainBaseTarget = pd.read_csv('../preprocessdata/pre_shuffled_target.csv')
    trainBase = pd.read_csv('../preprocessdata/pre_departition_train.csv')
    trainBaseWeight = trainBase['var11']
    #test = pd.read_csv('../data/pre_shuffled_test.csv')

    columns = trainBase.columns.values.tolist()
    columnsHighScore = trainBase.columns.values.tolist()


    print(trainBase.columns)
    
    trainBase = np.nan_to_num(np.array(trainBase))
    targetBase = np.nan_to_num(np.array(trainBaseTarget))
    #test = np.nan_to_num(np.array(test))
    
    gc.collect()   
   
    
    avg = 0
    avgLast = -1
    NumFolds = 5 


    clf = Lasso(alpha=0.00010) # found with tune_lasso.py

    
    
    
    print ("Data size: " + str(len(trainBase)))
    print("Begin Training")
    
    lenTrainBase = len(trainBase)
   

    gc.collect()
    
    
    featuresRemaining = []
    avgScore = []    
    
    
    while True:
        print(clf)
        avg = 0
    
        coef_dataset = np.zeros((len(columns),NumFolds))
   
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
            

            #clf.fit(train, target, sample_weight = weight
            clf.fit(train, target)
            predicted = clf.predict(trainTest) 
 
  
            print(str(score.normalized_weighted_gini(targetTest.ravel(), predicted.ravel(), weightTest.ravel())))
            avg += score.normalized_weighted_gini(targetTest.ravel(), predicted.ravel(), weightTest.ravel())/NumFolds

                 
            coef_dataset[:, foldCount] = clf.coef_                 

            foldCount = foldCount + 1
        

     
        
        coefs = coef_dataset.mean(1)        
        sorted_coefs = sorted(map(abs, coefs)) # must start by removing coefficients closest to zero.
        print(coefs)
        print("len coefs: " + str(len(sorted_coefs)))
        if len(sorted_coefs) < 5 :
            break
        
        threshold = sorted_coefs[5]

        print(str(len(columns)))
        print(trainBase.shape)
        
        toDrop = []        
        
        # hey, cannot drop var11 and id columns          
        for index in range(len(coefs) - 1, -1, -1): # must reverse columns all shift to lower numbers.
            if  abs(coefs[index]) <= threshold and columns[index] != "var11" and columns[index] != "id":# abs(), remove closest to zero.
                print("Drop: " + str(index) + " " + columns[index] + " " + str(coefs[index]))
                #trainBase = np.delete(trainBase,[index], axis=1)
                toDrop.append(index)
               
               
                #print(columns)
                if columns[index] in columns: 
                    columns.remove(columns[index])  
                #print(columns)
        
        print("start drop")
        trainBase = np.delete(trainBase,toDrop, axis=1)      
        print("End drop")        
        
        
        if avg > avgLast:
            print("Saving Copy " + str(avgLast) + " " + str(avg))
            avgLast = avg
            columnsHighScore = columns.copy()

        print("Threshold: " + str(threshold))        
        print ("------------------------Average: " + str(avg))
        print(columnsHighScore)
        print(str(len(columns)))
        print(trainBase.shape)
           
           
        featuresRemaining.append(len(columns))           
        avgScore.append(avg)
           
        #break
    
    
               
    gc.collect()    
    trainBase = pd.read_csv('../preprocessdata/pre_departition_train.csv')
    trainBase = trainBase.loc[:,columnsHighScore]
    trainBase.to_csv("../models/" + str(clf)[:5] +  "_train.csv", index = False)
    
    
    gc.collect()
    test = pd.read_csv('../preprocessdata/pre_departition_test.csv')
    test = test.loc[:,columnsHighScore]
    test.to_csv("../models/" + str(clf)[:5] + "_test.csv", index = False)  
      
      
    print(columnsHighScore)      
    print(featuresRemaining)
    print(avgScore)
    
    
if __name__=="__main__":
    run_stack(448)
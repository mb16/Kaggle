#!/usr/bin/env python

from sklearn import svm
import csv_io
import score
import math
from math import log
from sklearn import cross_validation

from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier

import numpy as np
import pandas as pd


import gc

import datetime
import random

 
    
import score
    
def run_stack(SEED):

    model = "pre_gb"


    trainBaseTarget = pd.read_csv('../preprocessdata/pre_shuffled_target_class.csv')
    trainBase = pd.read_csv('../models/' + model + '_train.csv')
    trainBaseWeight = trainBase['var11']
    test = pd.read_csv('../models/' + model + '_test.csv')


    #trainBase = shuffle(trainBase, random_state = SEED)

    print(trainBase.columns)
    trainBaseID = trainBase['id']
    testID = test['id']    

    
    trainBase = np.nan_to_num(np.array(trainBase))
    targetBase = np.nan_to_num(np.array(trainBaseTarget))
    test = np.nan_to_num(np.array(test))
    
    
    avg = 0
    NumFolds = 5



    clfs = [
        #RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, min_density=None, compute_importances=None)
        GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)
    ]        
    
    
    
    print ("Data size: " + str(len(trainBase)) + " " + str(len(test)))
    dataset_blend_train = np.zeros((len(trainBase), len(clfs)))
    dataset_blend_test = np.zeros((len(test), len(clfs)))
    


    
    print("Begin Training")
    
    lenTrainBase = len(trainBase)
    lenTest = len(test)
    
    

    gc.collect()
    
    for ExecutionIndex, clf in enumerate(clfs):
        print(clf)
        avg = 0
    

            
        dataset_blend_test_set = np.zeros((lenTest, NumFolds))

        
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
            predicted = clf.predict_proba(trainTest) 
            #print(predicted[:,0])
            print(predicted)
            dataset_blend_train[test_index, ExecutionIndex] = predicted[:,1]#[:,0] #needed for Ridge

     
            #print(targetTest.shape)
            #print(prpredictedob.shape)
            #print(weightTest.shape)

            print(str(score.normalized_weighted_gini(targetTest.ravel(), predicted[:,1].ravel(), weightTest.ravel())))
            avg += score.normalized_weighted_gini(targetTest.ravel(), predicted[:,1].ravel(), weightTest.ravel())/NumFolds
            #print(str(score.normalized_weighted_gini(targetTest.ravel(), predicted[:,0], weightTest.ravel())))
            #avg += score.normalized_weighted_gini(targetTest.ravel(), predicted[:,0], weightTest.ravel())/NumFolds

                 
            predicted = clf.predict_proba(test)         
            dataset_blend_test_set[:, foldCount] = predicted[:,1]#[:,0] 
        
                
            foldCount = foldCount + 1
        
            #break
        
        
        dataset_blend_test[:,ExecutionIndex] = dataset_blend_test_set.mean(1)  
        
    
        now = datetime.datetime.now()
        #print dataset_blend_test_set.mean(1) 
        #csv_io.write_delimited_file_single("../predictions/Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_test_set.mean(1))
        
        submission = pd.DataFrame(np.zeros((len(testID), 2)), columns=['id', 'target'])
        submission['target'] = dataset_blend_test[:,ExecutionIndex]
        submission['id'] = testID
        submission.to_csv("../predictions/Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", index = False)
        
        
        #csv_io.write_delimited_file_single("../predictions/Target_Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_train[:,ExecutionIndex] )        
        
        submission = pd.DataFrame(np.zeros((len(trainBaseID), 2)), columns=['id', 'target'])
        submission['target'] = dataset_blend_train[:,ExecutionIndex]
        submission['id'] = trainBaseID
        submission.to_csv("../predictions/Target_Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", index = False)
        
        
        csv_io.write_delimited_file("../log/RunLog.csv", [now.strftime("%Y %m %d %H %M %S"), "AVG." , str(avg), str(clf), "Folds:", str(NumFolds), "Model", "", "", ""], filemode="a",delimiter=",")
        
        
        print ("------------------------Average: " + str(avg))

        #np.savetxt('temp/dataset_blend_train.txt', dataset_blend_train)

    return dataset_blend_train, dataset_blend_test
                            
    
    
if __name__=="__main__":
    run_stack(448)
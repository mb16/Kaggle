#!/usr/bin/env python

from sklearn import svm
import csv_io
import score
import math
from math import log
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
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
    
    
def run_stack(SEED):


    trainBaseTarget = pd.read_csv('../data/target.csv')
    trainBase = pd.read_csv('../data/pre_shuffled_train.csv')
    test = pd.read_csv('../data/pre_shuffled_test.csv')


    trainBase = shuffle(trainBase, random_state = SEED)


    trainBaseID = trainBase['id']
    trainBaseWeight = ['var11']
    trainBase = trainBase[['var11', 'var12', 'var13', 'var14', 'var15', 'var16', 'var17']]
    testID = test['id']    
    testWeight = ['var11']
    test = test[['var11', 'var12', 'var13', 'var14', 'var15', 'var16', 'var17']]

    
    trainBase = np.nan_to_num(np.array(trainBase))
    targetBase = np.nan_to_num(np.array(trainBaseTarget))
    test = np.nan_to_num(np.array(test))
    
    
    avg = 0
    NumFolds = 5 


    predicted_list = []
    bootstrapLists = []

    #GradientBoostingRegressor(loss='ls', learning_rate=0.05, subsample=0.5, max_depth=10, n_estimators=30, random_state=166, min_samples_leaf=1),    

    clfs = [
       Ridge()
    ]        
    
    
    
    print ("Data size: " + str(len(trainBase)) + " " + str(len(test)))
    dataset_blend_train = np.zeros((len(trainBase), len(clfs)))
    dataset_blend_test = np.zeros((len(test), len(clfs)))
    

    trainNew = []
    trainTestNew = []
    testNew = []
    trainNewSelect = []
    trainTestNewSelect = []
    testNewSelect = []
    
    
    print("Begin Training")
    
    lenTrainBase = len(trainBase)
    lenTest = len(test)
    
    
    trainPre = []
    testPre = []
    
    gc.collect()
    
    for ExecutionIndex, clf in enumerate(clfs):
        print(clf)
        avg = 0
    
        predicted_list = []
            
        dataset_blend_test_set = np.zeros((lenTest, NumFolds))

        
        foldCount = 0

        Folds = cross_validation.KFold(lenTrainBase, n_folds=NumFolds, indices=True)
            
        for train_index, test_index in Folds:
    
            target = [targetBase[i] for i in train_index]
            train = [trainBase[i] for i in train_index]
            weight = [trainBaseWeight[i] for i in train_index]
            
            targetTest = [targetBase[i] for i in test_index]    
            trainTest = [trainBase[i] for i in test_index]    
            weightTest = [trainBaseWeight[i] for i in train_index]
            
            print()
            print ("Iteration: " + str(foldCount))
            #print "LEN: ", len(train), len(target)
            
            clf.fit(train, target)
            prob = clf.predict(trainTest) 
            
            dataset_blend_train[test_index, ExecutionIndex] = prob

     
     
            print(str(NumFolds * normalized_weighted_gini(targetTest, prob, weightTest)/NumFolds))
            avg += normalized_weighted_gini(targetTest, prob, weightTest)/NumFolds
                 
            predicted_probs = clf.predict(test)         
            dataset_blend_test_set[:, foldCount] = predicted_probs #[0]
        
                
            foldCount = foldCount + 1
        
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
        
        
        csv_io.write_delimited_file("../predictions/RunLog.csv", [now.strftime("%Y %m %d %H %M %S"), "AVG." , str(avg), str(clf), "Folds:", str(NumFolds), "Model", "", "", ""], filemode="a",delimiter=",")
        
        
        print ("------------------------Average: " + str(avg))

        #np.savetxt('temp/dataset_blend_train.txt', dataset_blend_train)

    return dataset_blend_train, dataset_blend_test
                            
    
    
if __name__=="__main__":
    run_stack(448)
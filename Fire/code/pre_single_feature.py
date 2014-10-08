#!/usr/bin/env python

from sklearn import svm
import csv_io
import score
import math
from math import log
from sklearn import cross_validation

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import scipy

import gc

import datetime
import random

from sklearn import preprocessing
from sklearn.utils import shuffle    

import score    
    
def run_pre_single_feature(SEED):



    trainBaseTarget = pd.read_csv('../data/pre_shuffled_target.csv')
    trainBase = pd.read_csv('../data/pre_shuffled_train.csv')
    trainBaseWeight = trainBase['var11']
    #test = pd.read_csv('../data/pre_shuffled_test.csv')


    columns = trainBase.columns


    print(trainBase.columns)
    trainBaseID = trainBase['id']

    
    trainBase = np.nan_to_num(np.array(trainBase))
    targetBase = np.nan_to_num(np.array(trainBaseTarget))
    
    
    avg = 0
    NumFolds = 5 


    predicted_list = []
    bootstrapLists = []

      
    
    
    
    #print ("Data size: " + str(len(trainBase)) + " " + str(len(test)))
    

    trainNew = []
    trainTestNew = []
    testNew = []
    trainNewSelect = []
    trainTestNewSelect = []
    testNewSelect = []
    
    
    print("Begin Training")
    
    lenTrainBase = len(trainBase)
    #lenTest = len(test)
    
    
    gc.collect()
    
    columnScores = {}    
    
    columnCount = 0    
    
    clfs = [
        #Ridge(),
        #RandomForestRegressor(n_estimators=30, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, min_density=None, compute_importances=False) ,
        #GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0),
        #AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None),
        SVR(kernel='rbf', degree=3, gamma=0.0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, probability=False, cache_size=200, verbose=False, max_iter=-1, random_state=None),
        # floating point over/under flow SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False, rho=None),
        #BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False),
    ]    
    
    
    
    for ExecutionIndex, clf in enumerate(clfs):
    
        print(clf)
        for column in range(0,trainBase.shape[1]):     
    
    
            avg = 0
        
            
            foldCount = 0
    
            Folds = cross_validation.KFold(lenTrainBase, n_folds=NumFolds, indices=True)
                
            for train_index, test_index in Folds:
        
                target = [targetBase[i] for i in train_index]
                train = [trainBase[i,column] for i in train_index]
                weight = [trainBaseWeight[i] for i in train_index]
                
                targetTest = [targetBase[i] for i in test_index]    
                trainTest = [trainBase[i,column] for i in test_index]    
                weightTest = [trainBaseWeight[i] for i in test_index]
                
                #print()
                #print ("Iteration: " + str(foldCount))
                #print "LEN: ", len(train), len(target)
                
                
                target = np.array(np.reshape(target, (-1, 1)) )           
                train = np.array(np.reshape(train, (-1, 1))  ) 
                weight = np.array(np.reshape(weight, (-1, 1)))              
    
                targetTest = np.array(np.reshape(targetTest, (-1, 1)) )  
                trainTest = np.array(np.reshape(trainTest, (-1, 1)) )  
                weightTest = np.array(np.reshape(weightTest, (-1, 1)))   
                
                #print(target.shape)
              
                
                clf.fit(train, target)
                prob = clf.predict(trainTest) 
          
                #print(targetTest)
                #print(prob)
                #print(weightTest)        
          
                #print(targetTest.shape)
                #print(prob.shape)
                #print(weightTest.shape)  
         
                print(str(score.normalized_weighted_gini(targetTest.ravel(), prob.ravel(), weightTest.ravel())))
                avg += score.normalized_weighted_gini(targetTest.ravel(), prob.ravel(), weightTest.ravel())/NumFolds
                     
                    
                foldCount = foldCount + 1
            
        
            
            print (str(columns[column]) + " Average Score: " + str(avg))
            columnScores[str(columns[column])] = avg
                
         
         
            columnCount = columnCount + 1
            if columnCount > 2:
                break     
         
         
        submission = pd.Series(columnScores)
        submission.to_csv("../featureanalysis/single_feaure_" + str(clf)[:5] + ".csv")   



if __name__=="__main__":
    run_pre_single_feature(448)
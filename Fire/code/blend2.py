#!/usr/bin/env python


import csv_io
import score

from sklearn import cross_validation
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


import gc

import datetime

 
import os

    
def run_stack(SEED):

    model = "Lasso"
    lossThreshold = 0.38

    trainBaseTarget = pd.read_csv('../preprocessdata/pre_shuffled_target.csv')
    trainBaseOrig = pd.read_csv('../models/' + model + '_train.csv')
    trainBaseWeight = trainBaseOrig['var11']
    testOrig = pd.read_csv('../models/' + model + '_test.csv')


    targetBase = np.nan_to_num(np.array(trainBaseTarget))


    trainBaseID = trainBaseOrig['id']
    testID = testOrig['id']    

    
    avg = 0
    NumFolds = 5


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


  
    clfs = [
        #GradientBoostingRegressor(loss='ls', learning_rate=0.05, subsample=0.5, max_depth=10, n_estimators=30, random_state=166, min_samples_leaf=1),   
        Lasso(alpha=0.000016681005372000593),
        #Ridge(),
        #LinearRegression(fit_intercept=True, normalize=False, copy_X=True)     
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
            predicted = clf.predict(trainTest) 
            #print(predicted[:,0])
            print(predicted)
            dataset_blend_train[test_index, ExecutionIndex] = predicted#[:,0] #needed for Ridge

     
            #print(targetTest.shape)
            #print(prpredictedob.shape)
            #print(weightTest.shape)

            print(str(score.normalized_weighted_gini(targetTest.ravel(), predicted.ravel(), weightTest.ravel())))
            avg += score.normalized_weighted_gini(targetTest.ravel(), predicted.ravel(), weightTest.ravel())/NumFolds
            #print(str(score.normalized_weighted_gini(targetTest.ravel(), predicted[:,0], weightTest.ravel())))
            #avg += score.normalized_weighted_gini(targetTest.ravel(), predicted[:,0], weightTest.ravel())/NumFolds

                 
            predicted = clf.predict(test)         
            dataset_blend_test_set[:, foldCount] = predicted#[:,0] 
        
                
            foldCount = foldCount + 1
        
            #break
        
        
        dataset_blend_test[:,ExecutionIndex] = dataset_blend_test_set.mean(1)  
        
    
        now = datetime.datetime.now()
        #print dataset_blend_test_set.mean(1) 
        #csv_io.write_delimited_file_single("../predictions/Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_test_set.mean(1))
        
        submission = pd.DataFrame(np.zeros((len(testID), 2)), columns=['id', 'target'])
        submission['target'] = dataset_blend_test[:,ExecutionIndex]
        submission['id'] = testID
        submission.to_csv("../submission/Blend_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", index = False)
        
        
        #csv_io.write_delimited_file_single("../predictions/Target_Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_train[:,ExecutionIndex] )        
        
        submission = pd.DataFrame(np.zeros((len(trainBaseID), 2)), columns=['id', 'target'])
        submission['target'] = dataset_blend_train[:,ExecutionIndex]
        submission['id'] = trainBaseID
        submission.to_csv("../submission/Target_Blend_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", index = False)
        
        
        csv_io.write_delimited_file("../log/RunLogBlend.csv", [now.strftime("%Y %m %d %H %M %S"), "AVG." , str(avg), str(clf), "Folds:", str(NumFolds), "Model", "Blend", "Stacks: ", stackFiles], filemode="a",delimiter=",")
        
        
        print ("------------------------Average: " + str(avg))

        #np.savetxt('temp/dataset_blend_train.txt', dataset_blend_train)

    return dataset_blend_train, dataset_blend_test
                            
    
    
if __name__=="__main__":
    run_stack(448)
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
    
def run_stack(SEED, col, alpha):


    model = "Lasso"
    lossThreshold = 0.46

    trainBaseTarget = pd.read_csv('../preprocess/pre_shuffled_target_' + col + '.csv')
    trainBaseOrig = pd.read_csv('../models/' + model + dset + '_train_' + col + '.csv')
    testOrig = pd.read_csv('../models/' + model + dset + '_test_' + col + '.csv')

    targetBase = np.nan_to_num(np.array(trainBaseTarget))

    #print(trainBase.columns)
    trainBaseID = trainBaseOrig['PIDN']
    testID = testOrig['PIDN']  
  
    
    avg = 0
    NumFolds = 5

	# ----------------------
	
    stackFiles = []
    for filename in os.listdir("../predictions"):
        parts = filename.split("_")
        if ( filename[0:5] == "Stack" and "Lasso" in filename  and float(parts[2]) < lossThreshold): # and "Lasso" in filename 

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

    clfs = [       
        Lasso(alpha=alpha),       
    ]        
    
    
    
    print ("Data size: " + str(len(trainBase)) + " " + str(len(test)))
    dataset_blend_train = np.zeros((len(trainBase), len(clfs)))
    dataset_blend_test = np.zeros((len(test), len(clfs)))
    
    averageSet = []

    
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

            
            targetTest = [targetBase[i] for i in test_index]    
            trainTest = [trainBase[i] for i in test_index]    

            

            #print "LEN: ", len(train), len(target)
            
            
            target = np.array(np.reshape(target, (-1, 1)) )           
           
    
            targetTest = np.array(np.reshape(targetTest, (-1, 1)) )  
          
            

            #clf.fit(train, target, sample_weight = weight
            clf.fit(train, target)
            predicted = clf.predict(trainTest) 
            #print(predicted[:,0])
            #print(predicted)
            dataset_blend_train[test_index, ExecutionIndex] = predicted#[:,0] #needed for Ridge

     
            #print(targetTest.shape)
            #print(prpredictedob.shape)
            #print(weightTest.shape)


            print(str(math.sqrt(mean_squared_error(targetTest, predicted))))
            avg += math.sqrt(mean_squared_error(targetTest, predicted))/NumFolds
                 
            predicted = clf.predict(test)         
            dataset_blend_test_set[:, foldCount] = predicted#[:,0] 
        
                
            foldCount = foldCount + 1
        
            #break
        

        
        averageSet.extend([avg])        


        
        dataset_blend_test[:,ExecutionIndex] = dataset_blend_test_set.mean(1)  
        
    
        now = datetime.datetime.now()
        #print dataset_blend_test_set.mean(1) 
        #csv_io.write_delimited_file_single("../predictions/Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_test_set.mean(1))
        
        submission = pd.DataFrame(np.zeros((len(testID), 2)), columns=['PIDN', col])
        submission[col] = dataset_blend_test[:,ExecutionIndex]
        submission['PIDN'] = testID
        submission.to_csv("../submission/temp/Blend_" + dset + "_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + "_" + col + ".csv", index = False)
        
        
        #csv_io.write_delimited_file_single("../predictions/Target_Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_train[:,ExecutionIndex] )        
        
        submission = pd.DataFrame(np.zeros((len(trainBaseID), 2)), columns=['PIDN', col])
        submission[col] = dataset_blend_train[:,ExecutionIndex]
        submission['PIDN'] = trainBaseID
        submission.to_csv("../submission/temp/Target_Blend_" + dset + "_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + "_" + col + ".csv", index = False)
        
        
        csv_io.write_delimited_file("../log/partial/RunLogBlend.csv", [now.strftime("%Y %m %d %H %M %S"), "AVG." , str(avg), str(clf), "Folds:", str(NumFolds), "Model", "Blend", "Stacks", stackFiles], filemode="a",delimiter=",")
        
        
        print ("------------------------Average: " + str(avg))

        #np.savetxt('temp/dataset_blend_train.txt', dataset_blend_train)

    return dataset_blend_train, dataset_blend_test, averageSet, clfs, NumFolds, model
                            
    
    
if __name__=="__main__":
    
    colsTarget = ['Ca','P','pH','SOC','Sand']     
    #alphas = [0.00167547491892,0.00167547491892,0.000452035365636,0.000699564215671,0.000452035365636]  # for base data set

    #alphas = [0.000699564215671,0.00621016941892,0.000699564215671,0.000121957046016,0.00108263673387]  # for data set 2
    alphas = [0.000193069772888,0.00931939576234,0.000390693993705,0.00159985871961,1e-08]  # for data set 3
    #alphas = [0.00159985871961,0.000555773658649,9.5409547635e-05,1e-08,4.71486636346e-05]  # for data set 4
    
    clfs = []
    averageSet = []
    dataset_blend_trainSet = []
    dataset_blend_testSet = []
    NumFolds = 0
    model = "" 
     
    for Index, col in enumerate(colsTarget):    
        dataset_blend_train, dataset_blend_test, avgs, clfs, NumFolds, model = run_stack(448,col, alphas[Index])
        averageSet.append(avgs)
        dataset_blend_trainSet.append(dataset_blend_train)
        dataset_blend_testSet.append(dataset_blend_test)




    trainBase = pd.read_csv('../models/' + model + '_train_' + colsTarget[0] + '.csv')
    test = pd.read_csv('../models/' + model + '_test_' +  colsTarget[0] + '.csv')
    trainBaseID = trainBase['PIDN']
    testID = test['PIDN']  

        
        
    print(averageSet)
    
    for ExecutionIndex, clf in enumerate(clfs): 
        average = 0
        
        submission1 = pd.DataFrame(np.zeros((len(testID), 6)), columns=['PIDN', 'Ca','P','pH','SOC','Sand'])        
        submission1['PIDN'] = testID      
      
        submission2 = pd.DataFrame(np.zeros((len(trainBaseID), 6)), columns=['PIDN', 'Ca','P','pH','SOC','Sand'])      
        submission2['PIDN'] = trainBaseID
        
        for idx, col in enumerate(colsTarget):       
            print(str(averageSet[idx][ExecutionIndex]))
            average += averageSet[idx][ExecutionIndex]
               
            submission1[col] = dataset_blend_testSet[idx][:,ExecutionIndex]        
        
            submission2[col] = dataset_blend_trainSet[idx][:,ExecutionIndex]   
            
            
        average = average/5    
        now = datetime.datetime.now()

        submission1.to_csv("../submission/Stack" + dset + "_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(average) + "_" + str(clf)[:12] + ".csv", index = False)
        
        submission2.to_csv("../submission/Target_Stack" + dset + "_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(average) + "_" + str(clf)[:12] + ".csv", index = False)
        
        
        csv_io.write_delimited_file("../log/RunLog.csv", [now.strftime("%Y %m %d %H %M %S"), "AVG." , str(average), str(clf), "Folds:", str(NumFolds), "Model", model, "dset", dset], filemode="a",delimiter=",")
               
               
               
        print ("------------------------Final Average: " + str(average))  
        
        
        
        
        
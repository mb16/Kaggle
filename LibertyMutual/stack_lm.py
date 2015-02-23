#!/usr/bin/env python

from sklearn import svm
import csv_io



from sklearn import cross_validation

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lasso, BayesianRidge,ElasticNet,SGDRegressor,ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
import numpy as np
import pandas as pd


import gc

import datetime

    
from sklearn.metrics import mean_squared_error
import math  
    

dset = "4"      
    
def run_stack(SEED, col, alpha, beta):


    model = "all"


    trainBaseTarget = pd.read_csv('../preprocess/pre_shuffled_target_' + col + '.csv')
    trainBase = pd.read_csv('../models/' + model + dset + '_train_' + col + '.csv')
    test = pd.read_csv('../models/' + model + dset + '_test_' + col + '.csv')


    print(trainBase.columns)
    trainBaseID = trainBase['PIDN']
    testID = test['PIDN']  



    colList = list(trainBase.columns)
    for item in colList:
        if  item.startswith('m'):
            print("Dropping: " + item)
            trainBase.drop([item], axis=1, inplace=True)
            test.drop([item], axis=1, inplace=True)




    trainBase.drop(['PIDN'], axis=1, inplace=True)
    test.drop(['PIDN'], axis=1, inplace=True)

    
    trainBase.drop(['BSAN'], axis=1, inplace=True)
    test.drop(['BSAN'], axis=1, inplace=True)

    trainBase.drop(['BSAS'], axis=1, inplace=True)
    test.drop(['BSAS'], axis=1, inplace=True)

    trainBase.drop(['BSAV'], axis=1, inplace=True)
    test.drop(['BSAV'], axis=1, inplace=True)

    trainBase.drop(['CTI'], axis=1, inplace=True)
    test.drop(['CTI'], axis=1, inplace=True)

    trainBase.drop(['ELEV'], axis=1, inplace=True)
    test.drop(['ELEV'], axis=1, inplace=True)
  
    trainBase.drop(['EVI'], axis=1, inplace=True)
    test.drop(['EVI'], axis=1, inplace=True)
    
    trainBase.drop(['LSTD'], axis=1, inplace=True)
    test.drop(['LSTD'], axis=1, inplace=True)
    
    trainBase.drop(['LSTN'], axis=1, inplace=True)
    test.drop(['LSTN'], axis=1, inplace=True)
    
    trainBase.drop(['REF1'], axis=1, inplace=True)
    test.drop(['REF1'], axis=1, inplace=True)

    trainBase.drop(['REF2'], axis=1, inplace=True)
    test.drop(['REF2'], axis=1, inplace=True)
    
    trainBase.drop(['REF3'], axis=1, inplace=True)
    test.drop(['REF3'], axis=1, inplace=True)
    
    trainBase.drop(['REF7'], axis=1, inplace=True)
    test.drop(['REF7'], axis=1, inplace=True)
    
    trainBase.drop(['TMAP'], axis=1, inplace=True)
    test.drop(['TMAP'], axis=1, inplace=True)
    
    trainBase.drop(['TMFI'], axis=1, inplace=True)
    test.drop(['TMFI'], axis=1, inplace=True)
    
    trainBase.drop(['0'], axis=1, inplace=True)
    test.drop(['0'], axis=1, inplace=True)
    
    trainBase.drop(['1'], axis=1, inplace=True)
    test.drop(['1'], axis=1, inplace=True)
    


    
    trainBase = np.nan_to_num(np.array(trainBase))
    targetBase = np.nan_to_num(np.array(trainBaseTarget))
    test = np.nan_to_num(np.array(test))
    
    
    avg = 0
    NumFolds = 5



    #GradientBoostingRegressor(loss='ls', learning_rate=0.05, subsample=0.5, max_depth=10, n_estimators=30, random_state=166, min_samples_leaf=1),    
        #Ridge()
    clfs = [
        #KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
        #SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False),
        #BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)
        #ElasticNet(alpha=0.00069956421567126271, l1_ratio=1/10, fit_intercept=True, normalize=False, precompute='auto', max_iter=10000, copy_X=True, tol=1/10000, warm_start=False, positive=False)        
        #LinearRegression(fit_intercept=True, normalize=False, copy_X=True),
        #BaggingRegressor(n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, n_jobs=1, random_state=None, verbose=0),
        #AdaBoostRegressor( n_estimators=1000, learning_rate=0.3, loss='linear', random_state=None)
        #Lasso(alpha=alpha),
        #ElasticNet(alpha=alpha)
        #SVR(C=10**alpha, kernel='rbf', degree=3, gamma=10**beta, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False)
        SVR(C=10**alpha, kernel='rbf', degree=3, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False)
        #Ridge(),
        #RandomForestRegressor(n_estimators=3000, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, min_density=None, compute_importances=None),
        
        
        #GradientBoostingRegressor(loss='ls', learning_rate=0.05, subsample=0.5, max_depth=10, n_estimators=15, random_state=166, min_samples_leaf=1),
        #GradientBoostingRegressor(loss='ls', learning_rate=0.05, subsample=0.5, max_depth=10, n_estimators=10, random_state=166, min_samples_leaf=1),
        #GradientBoostingRegressor(loss='ls', learning_rate=0.05, subsample=0.5, max_depth=5, n_estimators=15, random_state=166, min_samples_leaf=1),
        #GradientBoostingRegressor(loss='ls', learning_rate=0.05, subsample=0.5, max_depth=2, n_estimators=15, random_state=166, min_samples_leaf=1),
 
   
        #GradientBoostingRegressor(loss='ls', learning_rate=0.05, subsample=0.5, max_depth=10, n_estimators=100, random_state=166, min_samples_leaf=1),
        #GradientBoostingRegressor(loss='ls', learning_rate=0.05, subsample=0.5, max_depth=10, n_estimators=300, random_state=166, min_samples_leaf=1),
        #GradientBoostingRegressor(loss='ls', learning_rate=0.05, subsample=0.5, max_depth=10, n_estimators=1000, random_state=166, min_samples_leaf=1),
        #GradientBoostingRegressor(loss='ls', learning_rate=0.05, subsample=0.5, max_depth=10, n_estimators=30, random_state=166, min_samples_leaf=1),
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
            target = target.flatten() # use this for svm.
            targetTest = np.array(np.reshape(targetTest, (-1, 1)) )  
          
            #train = np.array(train)
          
            #print(train.shape)



            clf.fit(train, target.flatten())
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
        submission.to_csv("../predictions/partial/Stack" + dset + "_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + "_" + col + ".csv", index = False)
        
        
        #csv_io.write_delimited_file_single("../predictions/Target_Stack_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + ".csv", dataset_blend_train[:,ExecutionIndex] )        
        
        submission = pd.DataFrame(np.zeros((len(trainBaseID), 2)), columns=['PIDN', col])
        submission[col] = dataset_blend_train[:,ExecutionIndex]
        submission['PIDN'] = trainBaseID
        submission.to_csv("../predictions/partial/Target_Stack" + dset + "_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(avg) + "_" + str(clf)[:12] + "_" + col + ".csv", index = False)
        
        
        csv_io.write_delimited_file("../log/partial/RunLog.csv", [now.strftime("%Y %m %d %H %M %S"), "AVG." , str(avg), str(clf), "Folds:", str(NumFolds), "Model", model, "Column", col], filemode="a",delimiter=",")
        
        
        print ("------------------------Average: " + str(avg))

        #np.savetxt('temp/dataset_blend_train.txt', dataset_blend_train)

    return dataset_blend_train, dataset_blend_test, averageSet, clfs, NumFolds, model
                            
    
    
if __name__=="__main__":
    
    colsTarget = ['Ca','P','pH','SOC','Sand']     
    #alphas = [0.00167547491892,0.00167547491892,0.000452035365636,0.000699564215671,0.000452035365636]  # for base data set

    #alphas = [0.000699564215671,0.00621016941892,0.000699564215671,0.000121957046016,0.00108263673387]  # for data set 2
    alphas = [0.000923670857187,0.0148735210729,0.000923670857187,0.00017433288222,0.000923670857187]  # for data set 3
    alphas = [0.00159985871961,0.000555773658649,9.5409547635e-05,1e-08,4.71486636346e-05]  # for data set 4
    
    
    # Ca C=6 G=-4    
    # P C=1 G=-3      
    # pH C=2 G=-4      
    # SOC C=1 G=-4     
    # Sand C=6 G=-3      
    #alphas = [6,1,2,1,6]  # for data set 4
    alphas = [4,4,4,4,4]  # for data set 4
    #alphas = [3,3,3,3,3]  # for data set 4
    betas = [-4,-3,-4,-4,-3]  # for data set 4

    
    clfs = []
    averageSet = []
    dataset_blend_trainSet = []
    dataset_blend_testSet = []
    NumFolds = 0
    model = "" 
     
    for Index, col in enumerate(colsTarget):    
        dataset_blend_train, dataset_blend_test, avgs, clfs, NumFolds, model = run_stack(448,col, alphas[Index], betas[Index])
        averageSet.append(avgs)
        dataset_blend_trainSet.append(dataset_blend_train)
        dataset_blend_testSet.append(dataset_blend_test)




    trainBase = pd.read_csv('../models/' + model + dset + '_train_' + colsTarget[0] + '.csv')
    test = pd.read_csv('../models/' + model + dset + '_test_' +  colsTarget[0] + '.csv')
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

        submission1.to_csv("../predictions/Stack" + dset + "_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(average) + "_" + str(clf)[:12] + ".csv", index = False)
        
        submission2.to_csv("../predictions/Target_Stack" + dset + "_" + now.strftime("%Y%m%d%H%M%S") + "_" + str(average) + "_" + str(clf)[:12] + ".csv", index = False)
        
        
        csv_io.write_delimited_file("../log/RunLog.csv", [now.strftime("%Y %m %d %H %M %S"), "AVG." , str(average), str(clf), "Folds:", str(NumFolds), "Model", model, "dset", dset], filemode="a",delimiter=",")
               
               
               
        print ("------------------------Final Average: " + str(average))  
        
        
        
        
        


from sklearn.preprocessing import Imputer
import pandas as pd
import gc
import numpy as np
from scipy import stats
import datetime
from scipy.stats import itemfreq
from collections import Counter

def run_impute(SEED):
    
    partition = "A"    
    
    
    gc.collect() 
    
    trainBase = pd.read_csv('../preprocessdata/train2' + partition + '.csv')
    print("111")
    test = pd.read_csv('../preprocessdata/test2' + partition + '.csv')


    cols = [ 'var15A','varJ', 'varK', 'varL', 'varM', 'varN', 'varO',  'varP', 'varQ', 'varR', 'varS', 'varT','var13A', 'var16A', 'var16B' ]
    
    
    for Index, col in enumerate(cols):

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        print(col) 
        
        trainBaseArr = trainBase[col].values        
        testArr = test[col].values
        allVals = np.concatenate((trainBaseArr,testArr), axis=0)


 
        print(allVals) 
               
              
                
                
        if col == "var15A" or col == "var16B":
            
            indsTrain = np.where(np.isnan(trainBaseArr))      
            indsTest = np.where(np.isnan(testArr))               
                
            median = 0 # 16B all NA's get set to 0.

        elif col == "varP" or col == "varQ" or col == "varR" or col == "varS" or col == "varT" :
            
            trainBaseArr[trainBaseArr == 0.0]  = np.nan   
            testArr[testArr == 0.0]        = np.nan       
            
            continue 
 
        elif col == "var13A":
            indsTrain = np.where(np.isnan(trainBaseArr))      
            indsTest = np.where(np.isnan(testArr))             
            
            print("var13A " +  str(allVals[13]))
            median = allVals[13]
            
        elif col == "var16A":
            
            indsTrain = np.where(np.isnan(trainBaseArr))      
            indsTest = np.where(np.isnan(testArr))             
            
            trainBaseArr = trainBaseArr.astype('str')
            testArr = testArr.astype('str')            
            
            allVals = np.concatenate((trainBaseArr,testArr), axis=0)#.astype('str')
            print(allVals.astype('str'))            
            counts = Counter(allVals)
            print(counts)

            median = '0'
            maximum = 0
            for key in counts.keys():
                if counts[key] > maximum and key != "nan":
                    maximum = counts[key]
                    median = key

            print(median)
            
        elif col == 'varJ' or col == 'varK' or col == 'varL' or col == 'varM' or col == 'varN' or col == 'varO':

            trainBaseArr = trainBaseArr.astype('str')
            testArr = testArr.astype('str')

            indsTrain = np.where(np.char.find(trainBaseArr, 'Z') > -1)
            indsTest = np.where(np.char.find(testArr, 'Z') > -1)              

            allVals = np.concatenate((trainBaseArr,testArr), axis=0)#.astype('str')
            print(allVals.astype('str'))            
            counts = Counter(allVals)
            print(counts)

            median = 'Z'
            maximum = 0
            for key in counts.keys():
                if counts[key] > maximum and key != "Z" and key != "nan":
                    maximum = counts[key]
                    median = key

            print(median)
  
            
        else:
            
            indsTrain = np.where(np.isnan(trainBaseArr))      
            indsTest = np.where(np.isnan(testArr))             
            
            counts = Counter(allVals)
            print(counts)

            median = 0
            maximum = 0
            for key in counts.keys():
                if counts[key] > maximum and key != "nan":
                    maximum = counts[key]
                    median = key

            print(median)
        
        print("median: " + str(median))
        
        print(trainBaseArr) 
     
     
        trainBaseArr[indsTrain] = median  
        testArr[indsTest] = median

        
       
        if col == "varJ":
            for index, val in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']):
               
               indsTrain = np.where(np.char.find(trainBaseArr, val) > -1)
               indsTest = np.where(np.char.find(testArr, val) > -1)  
               
               trainBaseArr[indsTrain] = index  
               testArr[indsTest] = index        
        
        
        trainBase[col] = trainBaseArr          
        test[col] = testArr          
        
     
        gc.collect()    
    
    submission = pd.DataFrame(trainBase, columns = trainBase.columns)
    gc.collect()
    submission.to_csv("../preprocessdata/pre_first_train" + partition + ".csv", index = False)
    gc.collect()


    submission = pd.DataFrame(test, columns = test.columns)
    gc.collect()
    submission.to_csv("../preprocessdata/pre_first_test" + partition + ".csv", index = False)        


if __name__=="__main__":
    run_impute(448)

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from scipy.stats import mode

def run_onehot(SEED):
    

    dset = "5"

    trainBase = pd.read_csv('../data/training' + dset + '.csv')
    test = pd.read_csv('../data/sorted_test' + dset + '.csv')  

    
    cols = ['Depth']    
    

    for Index, col in enumerate(cols):

        print(col)

        trainBase[col] =  trainBase[col].astype(str)
        test[col] =  test[col].astype(str)

        # get data as numpy array
        trainBaseArr = trainBase[col].values
        testArr = test[col].values

    
        
        print(trainBaseArr)
        print(testArr)
        print(trainBaseArr.shape)
        print(testArr.shape)
        print(np.concatenate((trainBaseArr,testArr), axis=0))

        le = preprocessing.LabelEncoder()
        le.fit(np.concatenate((trainBaseArr,testArr), axis=0))

        trainBase[col] = le.transform(trainBaseArr)
        test[col] = le.transform(testArr)    
        

    
        
    colsTarget = ['Ca','P','pH','SOC','Sand']     

    for Index, col in enumerate(colsTarget):
    
        submission = pd.DataFrame(trainBase[col])  
        submission.to_csv("../preprocess/target_" + col + ".csv", index = False)
        trainBase.drop([col], axis=1, inplace=True)


    
    enc = OneHotEncoder()
    #print(np.concatenate((trainBase[cols].values,test[cols].values), axis=0))
    enc.fit(np.concatenate((trainBase[cols].values,test[cols].values), axis=0))
    
    newTrainBase = enc.transform(trainBase[cols].values).toarray()
    newTest = enc.transform(test[cols].values).toarray()


    trainBase.drop(cols, axis=1, inplace=True)
    test.drop(cols, axis=1, inplace=True)



    for col in range(newTrainBase.shape[1]):
        trainBase[col]  = newTrainBase[:,col]
        
    for col in range(newTest.shape[1]):
        test[col]  = newTest[:,col]        


    #trainBase = trainBase[:1000,]
    #test = test[:1000,]    
    

    submission = pd.DataFrame(trainBase) 
    submission.to_csv("../preprocess/pre_onehot_train" + dset + ".csv", index = False)


    submission = pd.DataFrame(test) 
    submission.to_csv("../preprocess/pre_onehot_test" + dset + ".csv", index = False)        


if __name__=="__main__":
    run_onehot(448)

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
        

    

    submission = pd.DataFrame(trainBase) 
    submission.to_csv("../preprocess/pre_onehot_train" + dset + ".csv", index = False)


    submission = pd.DataFrame(test) 
    submission.to_csv("../preprocess/pre_onehot_test" + dset + ".csv", index = False)        


if __name__=="__main__":
    run_onehot(448)
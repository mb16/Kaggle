
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from scipy.stats import mode

def run_onehot(SEED):
    

    trainBase = pd.read_csv('../preprocessdata/pre_first_trainA.csv')
    #trainBase = trainBase[:2000]

    test = pd.read_csv('../preprocessdata/pre_first_testA.csv')  
    #test = test[:2000]
    
    cols = ['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9']    
    
    #trainBase = trainBase[['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9']]
    #test = test[['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9']]    
    
    
    for Index, col in enumerate(cols):

        print(col)

        trainBase[col] =  trainBase[col].astype(str)
        test[col] =  test[col].astype(str)

        #print(np.array(trainBase[col]))

        # get data as numpy array
        trainBaseArr = trainBase[col].values
        testArr = test[col].values

        # find most frequent
        #u, indices = np.unique(trainBaseArr, return_inverse=True)
        #most_frequent = u[np.argmax(np.bincount(indices))]
        #print(most_frequent)

        # replace Z which stands for unknown
        #trainBaseArr[trainBaseArr[:, 0] == 'Z', 0] = most_frequent        
        #testArr[testArr[:, 0] == 'Z', 0] = most_frequent
        
        print(trainBaseArr)
        print(testArr)
        print(trainBaseArr.shape)
        print(testArr.shape)
        print(np.concatenate((trainBaseArr,testArr), axis=0))

        le = preprocessing.LabelEncoder()
        le.fit(np.concatenate((trainBaseArr,testArr), axis=0))

        trainBase[col] = le.transform(trainBaseArr)
        test[col] = le.transform(testArr)    
        

    
    
    
    submission = pd.DataFrame(trainBase['target'])  
    submission.to_csv("../data/target.csv", index = False)

    #submission = pd.DataFrame(trainBase['var11'])  
    #submission.to_csv("../data/weights.csv", index = False)
    
    
    trainBase.drop(['target'], axis=1, inplace=True)


    
    enc = OneHotEncoder()
    #print(np.concatenate((trainBase[cols].values,test[cols].values), axis=0))
    enc.fit(np.concatenate((trainBase[cols].values,test[cols].values), axis=0))
    
    newTrainBase = enc.transform(trainBase[cols].values).toarray()
    newTest = enc.transform(test[cols].values).toarray()


    trainBase.drop(cols, axis=1, inplace=True)
    test.drop(cols, axis=1, inplace=True)




    trainBase.drop(['dummy'], axis=1, inplace=True)
    test.drop(['dummy'], axis=1, inplace=True)
    


    for col in range(newTrainBase.shape[1]):
        trainBase[col]  = newTrainBase[:,col]
        
    for col in range(newTest.shape[1]):
        test[col]  = newTest[:,col]        


    #trainBase = trainBase[:1000,]
    #test = test[:1000,]    
    

    submission = pd.DataFrame(trainBase) 
    submission.to_csv("../preprocessdata/pre_onehot_trainA.csv", index = False)


    submission = pd.DataFrame(test) 
    submission.to_csv("../preprocessdata/pre_onehot_testA.csv", index = False)        


if __name__=="__main__":
    run_onehot(448)
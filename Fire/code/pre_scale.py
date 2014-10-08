

from sklearn.preprocessing import StandardScaler
import pandas as pd
import datetime
import numpy as np

def run_scale(SEED):
    
    partition = "A"   

    trainBase = pd.read_csv('../preprocessdata/pre_fourth_train' + partition + '.csv')
    test = pd.read_csv('../preprocessdata/pre_fourth_test' + partition + '.csv')    
        
    for Index, col in enumerate(test.columns):

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        print(col) 

        if col == "id" or col == "var11":
            continue
        
        trainBaseArr = trainBase[col].values        
        testArr = test[col].values
        allVals = np.concatenate((trainBaseArr,testArr), axis=0)

        scl = StandardScaler(copy=True, with_mean=True, with_std=True)
        scl.fit(allVals) # should fit on the combined sets.
        
        print("A")
        trainBase[col] = scl.transform(trainBase[col].values)
        print("B")        
        test[col] = scl.transform(test[col].values)
      
      
      
    trainBase.to_csv("../preprocessdata/pre_scaled_train" + partition + ".csv", index = False)      
    test.to_csv("../preprocessdata/pre_scaled_test" + partition + ".csv", index = False)       
    


if __name__=="__main__":
    run_scale(448)
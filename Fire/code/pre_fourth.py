

import pandas as pd
import gc
import numpy as np


def run_impute(SEED):
    
    partition = "A"    
    
    
    gc.collect() 
    
    trainBase = pd.read_csv('../preprocessdata/pre_imputed_train' + partition + '.csv')
    test = pd.read_csv('../preprocessdata/pre_imputed_test' + partition + '.csv')


    cols = [ 'crimeVar1', 'crimeVar2', 'crimeVar3', 'crimeVar4', 'crimeVar5', 'crimeVar6', 'crimeVar7', 'crimeVar8', 'crimeVar9' ]
    
    
    trainBaseTemp = np.array(trainBase[cols])
    testTemp = np.array(test[cols])    
    
    
    trainBase['crimeVarA'] = trainBaseTemp.min(1)
    trainBase['crimeVarB'] = trainBaseTemp.max(1)
    trainBase['crimeVarC'] = trainBaseTemp.mean(1)    
    test['crimeVarA'] = testTemp.min(1)
    test['crimeVarB'] = testTemp.max(1)
    test['crimeVarC'] = testTemp.mean(1)      
    
    gc.collect()    
    
    
    cols = ['geodemVar1', 'geodemVar2', 'geodemVar3', 'geodemVar4', 'geodemVar5', 'geodemVar6', 'geodemVar7', 'geodemVar8', 'geodemVar9', 'geodemVar10', 'geodemVar11', 'geodemVar12', 'geodemVar13', 'geodemVar14', 'geodemVar15', 'geodemVar16', 'geodemVar17', 'geodemVar18', 'geodemVar19', 'geodemVar20', 'geodemVar21', 'geodemVar22', 'geodemVar23', 'geodemVar24', 'geodemVar25', 'geodemVar26', 'geodemVar27', 'geodemVar28', 'geodemVar29', 'geodemVar30', 'geodemVar31', 'geodemVar32', 'geodemVar33', 'geodemVar34', 'geodemVar35', 'geodemVar36', 'geodemVar37' ]
    
    
    trainBaseTemp = np.array(trainBase[cols])
    testTemp = np.array(test[cols])    
    
    
    trainBase['geodemVarA'] = trainBaseTemp.min(1)
    trainBase['geodemVarB'] = trainBaseTemp.max(1)
    trainBase['geodemVarC'] = trainBaseTemp.mean(1)    
    test['geodemVarA'] = testTemp.min(1)
    test['geodemVarB'] = testTemp.max(1)
    test['geodemVarC'] = testTemp.mean(1)     
    
    
    gc.collect()
    trainBase.to_csv("../preprocessdata/pre_fourth_train" + partition + ".csv", index = False)
    gc.collect()


    test.to_csv("../preprocessdata/pre_fourth_test" + partition + ".csv", index = False)        

  
if __name__=="__main__":
    run_impute(448)


from sklearn.preprocessing import Imputer
import pandas as pd
import gc
import numpy as np
from scipy import stats
import datetime

def run_impute(SEED):
    
    partition = "A"       
    
    
    gc.collect() 
    
    trainBase = pd.read_csv('../preprocessdata/pre_onehot_train' + partition + '.csv')
    test = pd.read_csv('../preprocessdata/pre_onehot_test' + partition + '.csv')

    if partition == "A":
        cols = ['varP', 'varQ', 'varR', 'varS', 'varT','var10', 'var11', 'var12', 'var13', 'var14', 'var15', 'var16', 'var17', 'crimeVar1', 'crimeVar2', 'crimeVar3', 'crimeVar4', 'crimeVar5', 'crimeVar6', 'crimeVar7', 'crimeVar8', 'crimeVar9', 'geodemVar1', 'geodemVar2', 'geodemVar3', 'geodemVar4', 'geodemVar5', 'geodemVar6', 'geodemVar7', 'geodemVar8', 'geodemVar9', 'geodemVar10', 'geodemVar11', 'geodemVar12', 'geodemVar13', 'geodemVar14', 'geodemVar15', 'geodemVar16', 'geodemVar17', 'geodemVar18', 'geodemVar19', 'geodemVar20', 'geodemVar21', 'geodemVar22', 'geodemVar23', 'geodemVar24', 'geodemVar25', 'geodemVar26', 'geodemVar27', 'geodemVar28', 'geodemVar29', 'geodemVar30', 'geodemVar31', 'geodemVar32', 'geodemVar33', 'geodemVar34', 'geodemVar35', 'geodemVar36', 'geodemVar37']
    else:
        cols = ['weatherVar1', 'weatherVar2', 'weatherVar3', 'weatherVar4', 'weatherVar5', 'weatherVar6', 'weatherVar7', 'weatherVar8', 'weatherVar9', 'weatherVar10', 'weatherVar11', 'weatherVar12', 'weatherVar13', 'weatherVar14', 'weatherVar15', 'weatherVar16', 'weatherVar17', 'weatherVar18', 'weatherVar19', 'weatherVar20', 'weatherVar21', 'weatherVar22', 'weatherVar23', 'weatherVar24', 'weatherVar25', 'weatherVar26', 'weatherVar27', 'weatherVar28', 'weatherVar29', 'weatherVar30', 'weatherVar31', 'weatherVar32', 'weatherVar33', 'weatherVar34', 'weatherVar35', 'weatherVar36', 'weatherVar37', 'weatherVar38', 'weatherVar39', 'weatherVar40', 'weatherVar41', 'weatherVar42', 'weatherVar43', 'weatherVar44', 'weatherVar45', 'weatherVar46', 'weatherVar47', 'weatherVar48', 'weatherVar49', 'weatherVar50', 'weatherVar51', 'weatherVar52', 'weatherVar53', 'weatherVar54', 'weatherVar55', 'weatherVar56', 'weatherVar57', 'weatherVar58', 'weatherVar59', 'weatherVar60', 'weatherVar61', 'weatherVar62', 'weatherVar63', 'weatherVar64', 'weatherVar65', 'weatherVar66', 'weatherVar67', 'weatherVar68', 'weatherVar69', 'weatherVar70', 'weatherVar71', 'weatherVar72', 'weatherVar73', 'weatherVar74', 'weatherVar75', 'weatherVar76', 'weatherVar77', 'weatherVar78', 'weatherVar79', 'weatherVar80', 'weatherVar81', 'weatherVar82', 'weatherVar83', 'weatherVar84', 'weatherVar85', 'weatherVar86', 'weatherVar87', 'weatherVar88', 'weatherVar89', 'weatherVar90', 'weatherVar91', 'weatherVar92', 'weatherVar93', 'weatherVar94', 'weatherVar95', 'weatherVar96', 'weatherVar97', 'weatherVar98', 'weatherVar99', 'weatherVar100', 'weatherVar101', 'weatherVar102', 'weatherVar103', 'weatherVar104', 'weatherVar105', 'weatherVar106', 'weatherVar107', 'weatherVar108', 'weatherVar109', 'weatherVar110', 'weatherVar111', 'weatherVar112', 'weatherVar113', 'weatherVar114', 'weatherVar115', 'weatherVar116', 'weatherVar117', 'weatherVar118', 'weatherVar119', 'weatherVar120', 'weatherVar121', 'weatherVar122', 'weatherVar123', 'weatherVar124', 'weatherVar125', 'weatherVar126', 'weatherVar127', 'weatherVar128', 'weatherVar129', 'weatherVar130', 'weatherVar131', 'weatherVar132', 'weatherVar133', 'weatherVar134', 'weatherVar135', 'weatherVar136', 'weatherVar137', 'weatherVar138', 'weatherVar139', 'weatherVar140', 'weatherVar141', 'weatherVar142', 'weatherVar143', 'weatherVar144', 'weatherVar145', 'weatherVar146', 'weatherVar147', 'weatherVar148', 'weatherVar149', 'weatherVar150', 'weatherVar151', 'weatherVar152', 'weatherVar153', 'weatherVar154', 'weatherVar155', 'weatherVar156', 'weatherVar157', 'weatherVar158', 'weatherVar159', 'weatherVar160', 'weatherVar161', 'weatherVar162', 'weatherVar163', 'weatherVar164', 'weatherVar165', 'weatherVar166', 'weatherVar167', 'weatherVar168', 'weatherVar169', 'weatherVar170', 'weatherVar171', 'weatherVar172', 'weatherVar173', 'weatherVar174', 'weatherVar175', 'weatherVar176', 'weatherVar177', 'weatherVar178', 'weatherVar179', 'weatherVar180', 'weatherVar181', 'weatherVar182', 'weatherVar183', 'weatherVar184', 'weatherVar185', 'weatherVar186', 'weatherVar187', 'weatherVar188', 'weatherVar189', 'weatherVar190', 'weatherVar191', 'weatherVar192', 'weatherVar193', 'weatherVar194', 'weatherVar195', 'weatherVar196', 'weatherVar197', 'weatherVar198', 'weatherVar199', 'weatherVar200', 'weatherVar201', 'weatherVar202', 'weatherVar203', 'weatherVar204', 'weatherVar205', 'weatherVar206', 'weatherVar207', 'weatherVar208', 'weatherVar209', 'weatherVar210', 'weatherVar211', 'weatherVar212', 'weatherVar213', 'weatherVar214', 'weatherVar215', 'weatherVar216', 'weatherVar217', 'weatherVar218', 'weatherVar219', 'weatherVar220', 'weatherVar221', 'weatherVar222', 'weatherVar223', 'weatherVar224', 'weatherVar225', 'weatherVar226', 'weatherVar227', 'weatherVar228', 'weatherVar229', 'weatherVar230', 'weatherVar231', 'weatherVar232', 'weatherVar233', 'weatherVar234', 'weatherVar235', 'weatherVar236']
 
  
    
    for Index, col in enumerate(cols):

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        print(col) 
        
        trainBaseArr = trainBase[col].values        
        testArr = test[col].values
        allVals = np.concatenate((trainBaseArr,testArr), axis=0)
        allVals = allVals[~np.isnan(allVals)] # remove all nan's before finding median
        median = np.median(allVals, axis=0)
        print(median)
        
        
        inds = np.where(np.isnan(trainBaseArr))        
        trainBaseArr[inds] = median
        trainBase[col] = trainBaseArr        

 
        inds = np.where(np.isnan(testArr))        
        testArr[inds] = median
        test[col] = testArr  
        
        #print(trainBaseArr)        

        
        #print("A")
        # get data as numpy array
        #trainBaseArr = trainBase[col].values
        #trainBaseArr = np.matrix(trainBaseArr).T
        #print(trainBaseArr)
        #print(trainBaseArr.shape)
        #testArr = test[col].values
        #testArr = np.matrix(testArr).T
        #print("B")
        #imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=0, copy=True)         
        #print(np.matrix(np.concatenate((trainBase[col].values,test[col].values), axis=0)).T)
        #imp.fit(np.matrix(np.concatenate((trainBase[col].values,test[col].values), axis=0)).T)
        #print("C")
        #print(imp.transform(trainBaseArr).shape)
        #print(imp.transform(trainBaseArr))
        #print(len(imp.transform(trainBaseArr)))
        #print(trainBase[col])
        #print(len(trainBase[col]))
        #trainBase[col] = imp.transform(trainBaseArr).T
        #test[col] = imp.transform(testArr).T
        
        #print(col)    
        gc.collect()    
    
    submission = pd.DataFrame(trainBase, columns = trainBase.columns)
    gc.collect()
    submission.to_csv("../preprocessdata/pre_imputed_train" + partition + ".csv", index = False)
    gc.collect()


    submission = pd.DataFrame(test, columns = test.columns)
    gc.collect()
    submission.to_csv("../preprocessdata/pre_imputed_test" + partition + ".csv", index = False)        


if __name__=="__main__":
    run_impute(448)
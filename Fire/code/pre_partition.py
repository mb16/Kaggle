

from sklearn.preprocessing import Imputer
import pandas as pd
import gc
import numpy as np
from scipy import stats
import datetime

def run_impute(SEED):
    
    partition =  ['weatherVar1', 'weatherVar2', 'weatherVar3', 'weatherVar4', 'weatherVar5', 'weatherVar6', 'weatherVar7', 'weatherVar8', 'weatherVar9', 'weatherVar10', 'weatherVar11', 'weatherVar12', 'weatherVar13', 'weatherVar14', 'weatherVar15', 'weatherVar16', 'weatherVar17', 'weatherVar18', 'weatherVar19', 'weatherVar20', 'weatherVar21', 'weatherVar22', 'weatherVar23', 'weatherVar24', 'weatherVar25', 'weatherVar26', 'weatherVar27', 'weatherVar28', 'weatherVar29', 'weatherVar30', 'weatherVar31', 'weatherVar32', 'weatherVar33', 'weatherVar34', 'weatherVar35', 'weatherVar36', 'weatherVar37', 'weatherVar38', 'weatherVar39', 'weatherVar40', 'weatherVar41', 'weatherVar42', 'weatherVar43', 'weatherVar44', 'weatherVar45', 'weatherVar46', 'weatherVar47', 'weatherVar48', 'weatherVar49', 'weatherVar50', 'weatherVar51', 'weatherVar52', 'weatherVar53', 'weatherVar54', 'weatherVar55', 'weatherVar56', 'weatherVar57', 'weatherVar58', 'weatherVar59', 'weatherVar60', 'weatherVar61', 'weatherVar62', 'weatherVar63', 'weatherVar64', 'weatherVar65', 'weatherVar66', 'weatherVar67', 'weatherVar68', 'weatherVar69', 'weatherVar70', 'weatherVar71', 'weatherVar72', 'weatherVar73', 'weatherVar74', 'weatherVar75', 'weatherVar76', 'weatherVar77', 'weatherVar78', 'weatherVar79', 'weatherVar80', 'weatherVar81', 'weatherVar82', 'weatherVar83', 'weatherVar84', 'weatherVar85', 'weatherVar86', 'weatherVar87', 'weatherVar88', 'weatherVar89', 'weatherVar90', 'weatherVar91', 'weatherVar92', 'weatherVar93', 'weatherVar94', 'weatherVar95', 'weatherVar96', 'weatherVar97', 'weatherVar98', 'weatherVar99', 'weatherVar100', 'weatherVar101', 'weatherVar102', 'weatherVar103', 'weatherVar104', 'weatherVar105', 'weatherVar106', 'weatherVar107', 'weatherVar108', 'weatherVar109', 'weatherVar110', 'weatherVar111', 'weatherVar112', 'weatherVar113', 'weatherVar114', 'weatherVar115', 'weatherVar116', 'weatherVar117', 'weatherVar118', 'weatherVar119', 'weatherVar120', 'weatherVar121', 'weatherVar122', 'weatherVar123', 'weatherVar124', 'weatherVar125', 'weatherVar126', 'weatherVar127', 'weatherVar128', 'weatherVar129', 'weatherVar130', 'weatherVar131', 'weatherVar132', 'weatherVar133', 'weatherVar134', 'weatherVar135', 'weatherVar136', 'weatherVar137', 'weatherVar138', 'weatherVar139', 'weatherVar140', 'weatherVar141', 'weatherVar142', 'weatherVar143', 'weatherVar144', 'weatherVar145', 'weatherVar146', 'weatherVar147', 'weatherVar148', 'weatherVar149', 'weatherVar150', 'weatherVar151', 'weatherVar152', 'weatherVar153', 'weatherVar154', 'weatherVar155', 'weatherVar156', 'weatherVar157', 'weatherVar158', 'weatherVar159', 'weatherVar160', 'weatherVar161', 'weatherVar162', 'weatherVar163', 'weatherVar164', 'weatherVar165', 'weatherVar166', 'weatherVar167', 'weatherVar168', 'weatherVar169', 'weatherVar170', 'weatherVar171', 'weatherVar172', 'weatherVar173', 'weatherVar174', 'weatherVar175', 'weatherVar176', 'weatherVar177', 'weatherVar178', 'weatherVar179', 'weatherVar180', 'weatherVar181', 'weatherVar182', 'weatherVar183', 'weatherVar184', 'weatherVar185', 'weatherVar186', 'weatherVar187', 'weatherVar188', 'weatherVar189', 'weatherVar190', 'weatherVar191', 'weatherVar192', 'weatherVar193', 'weatherVar194', 'weatherVar195', 'weatherVar196', 'weatherVar197', 'weatherVar198', 'weatherVar199', 'weatherVar200', 'weatherVar201', 'weatherVar202', 'weatherVar203', 'weatherVar204', 'weatherVar205', 'weatherVar206', 'weatherVar207', 'weatherVar208', 'weatherVar209', 'weatherVar210', 'weatherVar211', 'weatherVar212', 'weatherVar213', 'weatherVar214', 'weatherVar215', 'weatherVar216', 'weatherVar217', 'weatherVar218', 'weatherVar219', 'weatherVar220', 'weatherVar221', 'weatherVar222', 'weatherVar223', 'weatherVar224', 'weatherVar225', 'weatherVar226', 'weatherVar227', 'weatherVar228', 'weatherVar229', 'weatherVar230', 'weatherVar231', 'weatherVar232', 'weatherVar233', 'weatherVar234', 'weatherVar235', 'weatherVar236']


    print("aaa")
    trainBase = pd.read_csv('../data/train2.csv')
    
    temp = trainBase[partition]  
    temp.to_csv("../preprocessdata/train2B.csv", index = False)
    gc.collect() 
    
    trainBase.drop(partition, axis=1, inplace=True)
    trainBase.to_csv("../preprocessdata/train2A.csv", index = False)
    gc.collect() 
    
    
    
    
    print("aaa")
    test = pd.read_csv('../data/test2.csv')
    
    temp = test[partition]  
    temp.to_csv("../preprocessdata/test2B.csv", index = False)
    gc.collect()     
    
    test.drop(partition, axis=1, inplace=True)
    test.to_csv("../preprocessdata/test2A.csv", index = False)
    gc.collect()     
    

if __name__=="__main__":
    run_impute(448)
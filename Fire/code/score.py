
from __future__ import division
import numpy as np
import pandas as pd


def gini(actual, pred, cmpcol = 0, sortcol = 1):
    print(str( len(actual)))
    print (str(len(pred)))
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
 
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


# implementation for Liberty Mutual Contest
def weighted_gini(act,pred,weight):
    df = pd.DataFrame({"act":act,"pred":pred,"weight":weight})  
    df = df.sort('pred',ascending=False) 
    df["random"] = (df.weight / df.weight.sum()).cumsum()
    total_pos = (df.act * df.weight).sum()
    df["cum_pos_found"] = (df.act * df.weight).cumsum()
    df["lorentz"] = df.cum_pos_found / total_pos
    #n = df.shape[0]
    #df["gini"] = (df.lorentz - df.random) * df.weight 
    #return df.gini.sum()
    gini = sum(df.lorentz[1:].values * (df.random[:-1])) - sum(df.lorentz[:-1].values * (df.random[1:]))
    return gini

def normalized_weighted_gini(act,pred,weight):
    return weighted_gini(act,pred,weight) / weighted_gini(act,act,weight)


# original working implementation.
#def weighted_gini(act,pred,weight):
#    df = pd.DataFrame({"act":act,"pred":pred,"weight":weight})    
#    df.sort('pred',ascending=False,inplace=True)        
#    df["random"] = (df.weight / df.weight.sum()).cumsum()
#    total_pos = (df.act * df.weight).sum()
#    df["cum_pos_found"] = (df.act * df.weight).cumsum()
#    df["lorentz"] = df.cum_pos_found / total_pos
#    df["gini"] = (df.lorentz - df.random) * df.weight  
#    return df.gini.sum()

#def normalized_weighted_gini(act,pred,weight):
#    return weighted_gini(act,pred,weight) / weighted_gini(act,act,weight)



#train = pd.read_csv('../data/train.csv')
#submission = pd.read_csv('../predictions/Target_Stack_20140728150729_0_Ridge(alpha=.csv')

#target = np.array(train['target'])
#prediction = np.array(submission['target'])


#print(str( gini_normalized(target, prediction)))


from sklearn.utils import shuffle
import pandas as pd
import shutil

def run_shuffle(SEED):
    
    target = pd.read_csv('../data/target.csv')
    trainBase = pd.read_csv('../preprocessdata/pre_sixth_train.csv')
    #test = pd.read_csv('../data/pre_scaled_test.csv')  # saves time to not read and write this since we do no processing.  
        
    
  
    
    newTrainBase, newTarget = shuffle(trainBase, target, random_state=SEED)


    submission = pd.DataFrame(newTrainBase, columns = trainBase.columns)
    submission.to_csv("../preprocessdata/pre_shuffled_train.csv", index = False)

    submission = pd.DataFrame(newTarget, columns = ['target'])  
    submission.to_csv("../preprocessdata/pre_shuffled_target.csv", index = False)

    # no need to shuffle test
    #test.to_csv("../data/pre_shuffled_test.csv", index = False)        

    shutil.copy2('../preprocessdata/pre_sixth_test.csv', "../data/pre_shuffled_test.csv")

if __name__=="__main__":
    run_shuffle(448)
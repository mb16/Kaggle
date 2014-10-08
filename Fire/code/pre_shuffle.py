

from sklearn.utils import shuffle
import pandas as pd
import shutil

def run_shuffle(SEED):

    weights = pd.read_csv('../data/weights.csv')
    newWeights = shuffle(weights, random_state=SEED)   
  
    submission = pd.DataFrame(newWeights, columns = ['weights'])  
    submission.to_csv("../preprocessdata/pre_shuffled_weights.csv", index = False)  

  
    
    target = pd.read_csv('../data/target.csv')
    newTarget = shuffle(target, random_state=SEED)   
  
    submission = pd.DataFrame(newTarget, columns = ['target'])  
    submission.to_csv("../preprocessdata/pre_shuffled_target.csv", index = False)  
    
            
    partition = "A"
  
    trainBase = pd.read_csv('../preprocessdata/pre_scaled_train' + partition + '.csv')
    newTrainBase = shuffle(trainBase, random_state=SEED)

    submission = pd.DataFrame(newTrainBase, columns = trainBase.columns)
    submission.to_csv("../preprocessdata/pre_shuffled_train" + partition + ".csv", index = False)

    shutil.copy2('../preprocessdata/pre_scaled_test' + partition + '.csv', "../preprocessdata/pre_shuffled_test" + partition + ".csv")



    partition = "B"
  
    trainBase = pd.read_csv('../preprocessdata/pre_scaled_train' + partition + '.csv')
    newTrainBase = shuffle(trainBase, random_state=SEED)

    submission = pd.DataFrame(newTrainBase, columns = trainBase.columns)
    submission.to_csv("../preprocessdata/pre_shuffled_train" + partition + ".csv", index = False)

    shutil.copy2('../preprocessdata/pre_scaled_test' + partition + '.csv', "../preprocessdata/pre_shuffled_test" + partition + ".csv")

if __name__=="__main__":
    run_shuffle(448)
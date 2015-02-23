

from sklearn.utils import shuffle
import pandas as pd
import shutil

def run_shuffle(SEED):

    dset = "5"
  
    colsTarget = ['Ca','P','pH','SOC','Sand']     

    for Index, col in enumerate(colsTarget):
        target = pd.read_csv('../preprocess/target_' + col + '.csv')
        newTarget = shuffle(target, random_state=SEED)   
      
        submission = pd.DataFrame(newTarget, columns = [col])  
        submission.to_csv("../preprocess/pre_shuffled_target_" + col + ".csv", index = False)  
        
            
  
    trainBase = pd.read_csv('../preprocess/pre_scaled_train' + dset + '.csv')
    newTrainBase = shuffle(trainBase, random_state=SEED)

    submission = pd.DataFrame(newTrainBase, columns = trainBase.columns)
    submission.to_csv("../preprocess/pre_shuffled_train" + dset + ".csv", index = False)

    shutil.copy2('../preprocess/pre_scaled_test' + dset + '.csv', "../preprocess/pre_shuffled_test" + dset + ".csv")


if __name__=="__main__":
    run_shuffle(448)


import pandas as pd
import datetime
import gc
import os


def run_sixth(SEED):
    

    A = open('../preprocessdata/pre_shuffled_trainA.csv', "r")
    B = open('../preprocessdata/pre_shuffled_trainB.csv', "r")

    out = open("../preprocessdata/pre_departition_train.csv",'w')

    lineA = A.readline()
    lineB = B.readline()

    # this uses very little memory, unlike concatenating pandas dataframes
    while True:

        out.write(lineA.rstrip() + "," + lineB)

        lineA = A.readline()
        lineB = B.readline()

        if not lineA or not lineB: break
        

    A.close()
    B.close()
    out.close()




    A = open('../preprocessdata/pre_shuffled_testA.csv', "r")
    B = open('../preprocessdata/pre_shuffled_testB.csv', "r")

    out = open("../preprocessdata/pre_departition_test.csv",'w')

    lineA = A.readline()
    lineB = B.readline()

    # this uses very little memory, unlike concatenating pandas dataframes
    while True:

        out.write(lineA.rstrip() + "," + lineB)

        lineA = A.readline()
        lineB = B.readline()

        if not lineA or not lineB: break
        

    A.close()
    B.close()
    out.close()



if __name__=="__main__":
    run_sixth(448)
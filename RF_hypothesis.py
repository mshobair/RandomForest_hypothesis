#!/usr/bin/env python
# coding: utf-8

# Author: Mahmoud Shobair
# Built: python v.3.6.9
# Updated: 30 January 2020


"""
The purpose of this python script is to estimate the predictive power of a specific set of features that have been previously
prioritized using a feature selection model.
We have been given a reference value of balanced accuracy 0.55.
Our goal is to assess whether applying these features in a random forest model can boost the performance metric.

Ths script allows for the user to shuffle the order of the input dataset to check if the splitting due to training
can impact model performance.
We limit the number of possible shuffling instances to 10.

In the virtual environment provided using pipenv, the user is prompted to enter a digit from 0 - 9.
An example of the output:

Please enter a digit (0 - 9) : 4
Balanced accuracy =  0.572
Positive space accuracy =  0.234
Balanced accuracy =  0.569
Positive space accuracy =  0.169
Balanced accuracy =  0.562
Positive space accuracy =  0.156
Balanced accuracy =  0.619
Positive space accuracy =  0.297
Balanced accuracy =  0.666
Positive space accuracy =  0.391
Average balanced accuracy =  0.597
Average positive accuracy =  0.249
"""


# load packages

from datetime import datetime
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


# shuffle input to obtain more splitting points

def shuffle_df(n):
    """
    Generate "n" randomly shuffled instance of input data frame :
    - in csv format & - named "input.csv"
    input  : input.csv
    output : tuple with size 2 containing:
             df_list_seed: list of shuffling seeds
             df_list: list of shuffled data frames
    """

    # read the input file
    df = pd.read_csv("input.csv")
    df_indexed = df.set_index('chemical_identifier')

    # generate random seed using current time
    now = datetime.now()
    initial_seed = int(now. strftime("%H%M%S%f"))
    seed = initial_seed // 1000000000

    # generate n data frames/seeds and add them to a list
    df_list = {}
    df_list_seed = {}
    for i in range(n):
        random_state_seed = seed + i
        df_shuffle = df_indexed.sample(frac = 1, # shuffle keeping total number of rows
                                       random_state = random_state_seed)
        df_list[i] = df_shuffle
        df_list_seed[i] = random_state_seed

    return df_list_seed, df_list


# pick one dataframe allowing user to enter key of dataframe
# user can only enter digits 0 - 9

## make this as a function  
df_n = -1 # could be any value outside of accepted range

tries = 0
while not  0 <= df_n <= 9:
    try:
        df_n = int(input("Please enter a digit (0 - 9) : "))
    except ValueError:
        print("wrong input format")
    tries += 1
    if tries > 3:
        print("Too many invalid inputs, exiting.....")
        sys.exit()

# passing user value to pick shuffled data set
input_shuffled = shuffle_df(10)[1][df_n] # picking n = 10 to shuffle 10 times


def split_prep(train_frac):
    """
    Prepare shuffled instance of input dataset for training and testing.

    input:    value of fraction of rows for training
    output:   tuple of lists of arrays in this order:
              hold_out (as external dataset, if model performance was optimized during testing)
              testing
              training
    """

    input_shuffled_bool = input_shuffled.astype(bool) # binary features and labels

    ## subsetting the training and testing sets in row-number ordering: hold_out, testing, training
    #  get the number of rows for testing to define the boudnds of testing/training rows
    test_end = int(input_shuffled_bool.shape[0] -
                   input_shuffled_bool.shape[0] * train_frac)

    #  split non_training rows in half into testing and hold_out
    withheld_end = test_end // 2

    ##### make a function to clean up try calling in panda
    #  get per group a list of numpy arrays of features and labels
    # hold_out
    hold_out = []
    hold_out_Y =  np.array(input_shuffled_bool.iloc[0 : withheld_end, 0])
    hold_out_X =  np.array(input_shuffled_bool.iloc[0 : withheld_end, 1:])
    hold_out.append(hold_out_X)
    hold_out.append(hold_out_Y)

    # testing
    testing = []
    testing_Y =  np.array(input_shuffled_bool.iloc[withheld_end : test_end, 0])
    testing_X =  np.array(input_shuffled_bool.iloc[withheld_end : test_end, 1:])
    testing.append(testing_X)
    testing.append(testing_Y)

    # training
    training = []
    training_Y = np.array(input_shuffled_bool.iloc[test_end : -1 , 0])
    training_X = np.array(input_shuffled_bool.iloc[test_end : -1 , 1:])
    training.append(training_X)
    training.append(training_Y)

    return hold_out, testing, training



### Model building
## Data prep
# Get train and test datasets with split ratio 8:1:1. Smaller training portions showed less predictive power.
# train
X_train = split_prep(0.8)[2][0]
Y_train = split_prep(0.8)[2][1]
# test
X_test = split_prep(0.8)[1][0]
Y_test = split_prep(0.8)[1][1]
# hold_out as external dataset
X_hold = split_prep(0.8)[0][0]
Y_hold = split_prep(0.8)[0][1]

## Algorithm : RandomForest was selected to act as null hypothesis tester
# Implementing 5-fold cross validation

skfolds5 = StratifiedKFold(n_splits = 5) # use stratified method to address label imbalance

# initializing and instantiating
rf = RandomForestClassifier()
Balanced_accuracy_CV = 0
Pos_accuracy_CV = 0

# loop through the splits and get train and test subsets
for train_index, test_index in skfolds5.split(X_train, Y_train):
    clone_rf = clone(rf)
    X_train_folds = X_train[train_index]
    y_train_folds = Y_train[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = Y_train[test_index]

    # train and test in the split clone and
    # print predictive performance: balanced accuracy and specifically True-label prediction
    clone_rf.fit(X_train_folds, y_train_folds)
    y_pred = clone_rf.predict(X_test_fold)
    tn, fp, fn, tp = confusion_matrix(y_test_fold, y_pred).flatten()
    # performance metrics
    BA = (tp/(tp+fn) + tn/(tn+fp))/2
    Balanced_accuracy_CV += BA
    PA = tp/(tp+fn)
    Pos_accuracy_CV += PA
    print("Balanced accuracy = ", round(BA, 3))
    print("Positive space accuracy = ", round(PA, 3))

print("Average balanced accuracy = ", round(Balanced_accuracy_CV/5, 3))
print("Average positive accuracy = ", round(Pos_accuracy_CV/5, 3))

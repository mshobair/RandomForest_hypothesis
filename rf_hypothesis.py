#!/usr/bin/env python
# coding: utf-8

# Author: Mahmoud Shobair
# Built: python v.3.6.9
# Updated: 30 January 2020


"""
The purpose of this python script is to estimate the predictive power
of a specific set of features that have been previously
prioritized using a feature selection model.
We have been given a reference value of balanced accuracy 0.55.
Our goal is to assess whether applying these features in a random forest
model can boost the performance metric.

Ths script allows for the user to shuffle the order of the input dataset
to check if the splitting due to training
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


def get_shuffled():
    """
    input:
    output:
    """
    # user can only enter digits 0 - 9
    def get_input():
        """
         pick one dataframe allowing user to enter key of dataframe
         user can only enter digits 0 - 9
        """

        n_df = -1 # could be any value outside of accepted range
        tries = 0
        while not  0 <= n_df <= 9:
            try:
                n_df = int(input("Please enter a digit (0 - 9) : "))
            except ValueError:
                print("wrong input format")
            tries += 1
            if tries > 3:
                print("Too many invalid inputs, exiting.....")
                sys.exit()
        return n_df

    n_df = get_input()


    def shuffle_df():
        """
        Generate "n_shuffles" randomly shuffled instance of input data frame :
        - in csv format & - named "input.csv"
        input  : input.csv
        output : tuple with size 2 containing:
                 df_list_seed: list of shuffling seeds
                 df_list: list of shuffled data frames
        """
        # read the input file
        df_input = pd.read_csv("input.csv")
        df_indexed = df_input.set_index('chemical_identifier')
        # generate random seed using current time
        now = datetime.now()
        initial_seed = int(now. strftime("%H%M%S%f"))
        seed = initial_seed // 1000000000
        # generate n data frames/seeds and add them to a list
        df_list = {}
        df_list_seed = {}
        for i in range(10):
            random_state_seed = seed + i
            df_shuffle = df_indexed.sample(frac=1, # shuffle keeping total number of rows
                                           random_state=random_state_seed)
            df_list[i] = df_shuffle
            df_list_seed[i] = random_state_seed

        return df_list_seed, df_list
        # passing user value to pick shuffled data set
    input_shuffled = shuffle_df()[1][n_df] # picking n = 10 to shuffle 10 times

    return input_shuffled




def split_prep(train_frac):
    """
    Prepare shuffled instance of input dataset for training and testing.

    input:    train_frac;value of fraction of rows for training
              input shuffled; shuffled instance of the data frame
    output:   tuple of lists of arrays in this order:
              hold_out (as external dataset, if model performance was optimized during testing)
              testing
              training
    """
    # passing user value to pick shuffled data set
    input_shuffled = get_shuffled()

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
    hold_out_y = np.array(input_shuffled_bool.iloc[0 : withheld_end, 0])
    hold_out_x = np.array(input_shuffled_bool.iloc[0 : withheld_end, 1:])
    hold_out.append(hold_out_x)
    hold_out.append(hold_out_y)

    # testing
    testing = []
    testing_y = np.array(input_shuffled_bool.iloc[withheld_end : test_end, 0])
    testing_x = np.array(input_shuffled_bool.iloc[withheld_end : test_end, 1:])
    testing.append(testing_x)
    testing.append(testing_y)

    # training
    training = []
    training_y = np.array(input_shuffled_bool.iloc[test_end : -1, 0])
    training_x = np.array(input_shuffled_bool.iloc[test_end : -1, 1:])
    training.append(training_x)
    training.append(training_y)

    return hold_out, testing, training




def get_ba():
    """
    input
    output

    """

    ### Model building
    ## Data prep
    # Get train and test datasets with split ratio 8:1:1.
    # Smaller training portions showed less predictive power.
    # train
    x_train = split_prep(0.8)[2][0]
    y_train = split_prep(0.8)[2][1]
    ## Algorithm : RandomForest was selected to act as null hypothesis tester
    # Implementing 5-fold cross validation

    skfolds5 = StratifiedKFold(n_splits=5) # use stratified method to address label imbalance

    # initializing and instantiating
    rf_c = RandomForestClassifier()
    balanced_accuracy_cv = 0
    pos_accuracy_cv = 0

    # loop through the splits and get train and test subsets
    for train_index, test_index in skfolds5.split(x_train, y_train):
        clone_rf = clone(rf_c)
        x_train_folds = x_train[train_index]
        y_train_folds = y_train[train_index]
        x_test_fold = x_train[test_index]
        y_test_fold = y_train[test_index]

        # train and test in the split clone and
        # print predictive performance: balanced accuracy and specifically True-label prediction
        clone_rf.fit(x_train_folds, y_train_folds)
        y_pred = clone_rf.predict(x_test_fold)
        tn_clone, fp_clone, fn_clone, tp_clone = confusion_matrix(y_test_fold, y_pred).flatten()
        # performance metrics
        ba_clone = (tp_clone/(tp_clone+fn_clone) + tn_clone/(tn_clone+fp_clone))/2
        balanced_accuracy_cv += ba_clone
        pa_clone = tp_clone/(tp_clone+fn_clone)
        pos_accuracy_cv += pa_clone
        print("Balanced accuracy = ", round(ba_clone, 3))
        print("Positive space accuracy = ", round(pa_clone, 3))

    print("Average balanced accuracy = ", round(balanced_accuracy_cv/5, 3))
    print("Average positive accuracy = ", round(pos_accuracy_cv/5, 3))
get_ba()

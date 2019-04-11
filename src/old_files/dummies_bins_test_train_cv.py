# Author: Mohit Gangwani
# Date: 11/05/2018
# Git-Hub: Data-is-Life

import pandas as pd
import os
import pickle
# import numpy as np
from random import randint
from sklearn.preprocessing import PolynomialFeatures as MMXs
from sklearn.model_selection import train_test_split


def save_pickle(something, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as fh:
        pickle.dump(something, fh, pickle.DEFAULT_PROTOCOL)


def initial_df():
    '''Input:
    file_name = name & location of the csv file

    Removes all the games where the result was a draw.

    Returns:
    df = initial dataframe
    df_len = length of dataframe'''

    df = pd.read_csv('../data/use_for_predictions.csv')
    for i in df.index:
        if df.loc[i, 'result'] == 0.5 and df.loc[i, 'color'] == 1.0:
            df.loc[i, 'result'] = 1.0 if df.loc[i, 'diff'] <= -30 else 0.0
        elif df.loc[i, 'result'] == 0.5 and df.loc[i, 'color'] == 0.0:
            df.loc[i, 'result'] = 1.0 if df.loc[i, 'diff'] <= 30 else 0.0

    df = df.loc[df['result'] != 0.5].copy()
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)

    return df

# df = initial_df()

def bin_df_get_y(df):
    '''Input:
    df = clean dataframe

    This creates y array, converts negative values to positive by taking the
    absolute of the minimum and adding it to all elo differences, and rounds
    elo, opp_elo, and difference to nearest 10.

    Returns:
    df = cleaner dataframe
    y = results as a 1D numpy array'''

    y = df['result'].values

    df.loc[:, 'diff'] = round(df['diff'] / 10) * 10
    df.loc[:, 'elo'] = round(df['elo'] / 10) * 10
    df.loc[:, 'opp_elo'] = round(df['opp_elo'] / 10) * 10
#    a = pd.get_dummies(df.weekday, prefix='wd', drop_first=True)
#    df = pd.concat([df, a], axis=1, sort=False)
    df.drop(columns=['result', 'day', 'weekday', 'day_game_num', 'color',
                     'start_time'], inplace=True)

#    df.rename(columns={'start_time': 'time', 'day_game_num': 'game_num'},
#              inplace=True)

    return df, y


def partial_df(df, perc, start_row_order, cv, rando):
    '''Input:
    df = clean dataframe
    perc = split the dataframe in fraction

    Using all the games will not produce the best results, since the players
    playing style and level of play changes over time. Hence, this is used to
    get a part of the dataframe.

    eg: perc = 1/4, len(df) = 2000.
    This will give last 25% (500 rows) of the dataframe.

    Returns:
    df_s = smaller dataframe'''

    if int(perc) == 1:
        return df

    DFLN = int(len(df) * perc)
    x = len(df) - DFLN - 1
    LNDF = len(df)

    if rando.lower() == 'y':
        start_row = randint(int(x*(cv-1)/cv), x)
        end_row = (start_row + DFLN)

        if end_row >= (len(df) - 1):
            df_s = df[start_row:].copy()
        else:
            df_s = df[start_row: (end_row + 1)].copy()

    elif start_row_order >= 0:
        start_row = int(((LNDF - DFLN) / cv) * start_row_order)
        end_row = (start_row + DFLN)
        if end_row >= (len(df) - 1):
            df_s = df[start_row:].copy()
        else:
            df_s = df[start_row: (end_row + 1)].copy()

    else:
        df_s = df[x:].copy()

    df_s.reset_index(inplace=True)
    df_s.drop(columns=['index'])

    return df_s


def xy_tt(df, splt):
    '''Input:
    X = array used for prediction
    y = results
    splt = desired split for X and y

    If a number less than 1 is given for split, the split is considered for
    training data percentage.
    If a number greater than 1 is given for split, the split is considered for
    number of test data samples.

    Example:
    Total # of samples = 1,000

    splt=0.90
    training data = 900 samples, test data = 100 samples

    splt=100
    training data = 900 samples, test data = 100 samples

    Returns:
    X_train = array to train
    X_test = array to test
    y_train = results to train
    y_test = results to test predictions
    X = array used for prediction'''

    df, y = bin_df_get_y(df)
    X = df.values

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=splt, shuffle=False)

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    SCaler = MMXs().fit(X_train)
    X_train = SCaler.transform(X_train)
    X_test = SCaler.transform(X_test)
    save_pickle(SCaler, '../data/scale.pickle')

    return X_train, X_test, y_train, y_test


# X_train, X_test, y_train, y_test, X = xy_tt(df, .1)


# def xy_custom(df, y, splt, cols):
#    '''Input:
#    df = cleaned dataframe
#    y = all result values in an Numpy Array
#    splt = Split size for test set in % as 0.80 or # as 200
#    cols = list of columns to create X values to predict over
#
#    This function creates X array, X_train, X_test, y_train, and y_test.
#    If the columns are not elo difference or color, it creates dummy columns.
#
#    Returns:
#    X = values to run predictions
#    X_train = training prediction set
#    X_test = testing prediction set
#    y_train = training result set
#    y_test = testing result set'''
#
#    X = df.values
#
#    if X.shape[1] <= 1:
#        X = X.reshape(-1, 1)
#
#    X_train, X_test, y_train, y_test = xy_tt(X, y, splt)
#
#    return X_train, X_test, y_train, y_test, X

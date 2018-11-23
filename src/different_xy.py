# Author: Mohit Gangwani
# Date: 11/21/2018
# Git-Hub: Data-is-Life

import pandas as pd
import numpy as np

from pandas import get_dummies as gd


def clean_df_y(df):
    '''
    Input: clean dataframe
    Output: cleaner dataframe, results as a 1D numpy array
    Creates bins for all differences. All bins labels are roughly calculated
    based on winning probability from original ELO equation:
    1/(1+10^m), where m = (elo difference)/400
    Also, bins the start time'''

    dif_bn = list(range(-1000, -600, 100))
    dif_bn.extend(list(range(-600, -250, 50)))
    dif_bn.extend(list(range(-250, -200, 25)))
    dif_bn.extend(list(range(-200, -100, 10)))
    dif_bn.extend(list(range(-100, 105, 5)))
    dif_bn.extend(list(range(110, 210, 10)))
    dif_bn.extend(list(range(225, 325, 25)))
    dif_bn.extend(list(range(350, 550, 50)))
    dif_bn.extend(list(range(600, 1100, 100)))

    dif_lbl = list(range(8))
    dif_lbl.extend(list(range(8, 23, 2)))
    dif_lbl.extend(list(range(23, 79)))
    dif_lbl.extend(list(range(80, 93, 2)))
    dif_lbl.extend(list(range(93, 100)))

    df.loc[:, 'diff_bin'] = pd.cut(df['diff'], bins=dif_bn, labels=dif_lbl)
    df.loc[:, 'time_bin'] = pd.cut(df['start_time'], bins=24, labels=False)

    y = np.array(df['result'])

    df.drop(columns=['result', 'opp_elo', 'elo', 'start_time', 'diff', 'day'],
            inplace=True)

    return df, y


def xy_tt(X, y, splt):
    '''
    Input:
    X = array used for prediction
    y = results
    splt = desired split for X and y

    Output:
    X_train = array to train
    X_test = array to test
    y_train = results to train
    y_test = results to test predictions
    X = array used for prediction

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
    '''

    if splt > 1:
        splitze = len(X) - int(splt)
    else:
        splitze = int(len(X) * splt)

    X_train = X[:splitze]
    y_train = y[:splitze]
    X_test = X[splitze:]
    y_test = y[splitze:]

    print(f'y Shape: {y.shape}')
    print(f'X Shape: {X.shape}')
    print(f'X_train Shape: {X_train.shape}')
    print(f'X_test Shape: {X_test.shape}')
    print(f'y_train Shape: {y_train.shape}')
    print(f'y_test Shape: {y_test.shape}')

    return X_train, X_test, y_train, y_test


def xy_custom(df, y, splt, cols):

    df_n = df[cols].copy()

    if len(cols) == 1:
        if cols == ['diff_bin'] or cols == ['color']:
            X = df_n.values
            X = X.reshape(-1, 1)
        else:
            df_n = gd(df_n, prefix='a', drop_first=True, columns=cols)
            X = df_n.values

    elif len(cols) == 2:
        if cols == ['diff_bin', 'color']:
            X = df_n.values
        elif (cols[0]=='diff_bin' or cols[0]=='color') and cols[1] != 'color':
            df_n = gd(df_n, prefix='a', drop_first=True, columns=[cols[1]])
            X = df_n.values
        else:
            df_n = gd(df_n, prefix='a', drop_first=True, columns=[cols[0]])
            df_n = gd(df_n, prefix='b', drop_first=True, columns=[cols[1]])
            X = df_n.values

    elif len(cols) == 3:
        if cols[0] == 'diff_bin' and cols[1] == 'color':
            df_n = gd(df_n, prefix='a', drop_first=True, columns=[cols[2]])
            X = df_n.values
        elif (cols[0]=='diff_bin' or cols[0]=='color') and cols[1] != 'color':
            df_n = gd(df_n, prefix='a', drop_first=True, columns=[cols[1]])
            df_n = gd(df_n, prefix='b', drop_first=True, columns=[cols[2]])
            X = df_n.values
        else:
            df_n = gd(df_n, prefix='a', drop_first=True, columns=[cols[0]])
            df_n = gd(df_n, prefix='b', drop_first=True, columns=[cols[1]])
            df_n = gd(df_n, prefix='c', drop_first=True, columns=[cols[2]])
            X = df_n.values

    elif len(cols) == 4:
        if cols[0] == 'diff_bin' and cols[1] == 'color':
            df_n = gd(df_n, prefix='a', drop_first=True, columns=[cols[2]])
            df_n = gd(df_n, prefix='a', drop_first=True, columns=[cols[3]])
            X = df_n.values
        else:
            df_n = gd(df_n, prefix='a', drop_first=True, columns=[cols[1]])
            df_n = gd(df_n, prefix='b', drop_first=True, columns=[cols[2]])
            df_n = gd(df_n, prefix='c', drop_first=True, columns=[cols[3]])
            X = df_n.values

    else:
        df_n = gd(df_n, prefix='a', drop_first=True, columns=[cols[2]])
        df_n = gd(df_n, prefix='b', drop_first=True, columns=[cols[3]])
        df_n = gd(df_n, prefix='c', drop_first=True, columns=[cols[4]])
        X = df_n.values

    X_train, X_test, y_train, y_test = xy_tt(X, y, splt)

    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')
    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')

    return X_train, X_test, y_train, y_test, X


'''Using 5 pre-game parameters to predict: 
   1. Elo difference (binned)
   2. Assigned color (1 for White, 0 for Black)
   3. Start time of the game (binned and split into dummy columns)
   4. Game time (split into dummy columns)
   5. Day of the week (split into dummy columns)
   
   There are 31 different combinations to use for predictions. Using
   numbers as the parameters are listed as their names:
   
   1, 2, 3, 4, 5 => 5 combos
   1+2, 1+3, 1+4, 1+5, 2+3, 2+4, 2+5, 3+4, 3+5, 4+5 => 10 combos
   1+2+3, 1+2+4, 1+2+5, 1+3+4, 1+3+5, 1+4+5, 2+3+4, 2+3+5, 2+4+5, 3+4+5 => 10 combos
   1+2+3+4, 1+2+3+5, 1+2+4+5, 1+3+4+5, 2+3+4+5 => 5 combos
   1+2+3+4+5 => 1 combination
   
   Ideally I would like to try all of them to find out which exact combination
   determines the most accurate result.

   Running 20 of these on Kaggle Kernel. Only ones that are not being tested
   are the ones that don't have ELO difference.
   '''

all_cols_try = [
    ['diff_bin'], ['color'], ['time_bin'], ['game_time'], ['weekday'],

    ['diff_bin', 'color'], ['diff_bin', 'time_bin'], ['diff_bin', 'game_time'],
    ['diff_bin', 'weekday'], ['color', 'time_bin'], ['color', 'game_time'],
    ['color', 'weekday'], ['time_bin', 'game_time'], ['time_bin', 'weekday'],
    ['game_time', 'weekday'],

    ['diff_bin', 'color', 'time_bin'], ['diff_bin', 'color', 'game_time'],
    ['diff_bin', 'color', 'weekday'], ['diff_bin', 'time_bin', 'game_time'],
    ['diff_bin', 'time_bin', 'weekday'], ['diff_bin', 'game_time', 'weekday'],
    ['color', 'time_bin', 'game_time'], ['color', 'time_bin', 'weekday'],
    ['color', 'game_time', 'weekday'], ['weekday', 'time_bin', 'game_time'],

    ['diff_bin', 'color', 'time_bin', 'game_time'],
    ['diff_bin', 'color', 'time_bin', 'weekday'],
    ['diff_bin', 'color', 'game_time', 'weekday'],
    ['diff_bin', 'time_bin', 'game_time', 'weekday'],
    ['color', 'weekday', 'time_bin', 'game_time'],

    ['diff_bin', 'color', 'game_time', 'time_bin', 'weekday']

]

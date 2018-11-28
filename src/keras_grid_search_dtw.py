# Author: Mohit Gangwani
# Date: 11/21/2018
# Git-Hub: Data-is-Life

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from pandas import get_dummies as gd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier
import keras
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
keras.backend.device_lib.list_local_devices(
    tf.ConfigProto(log_device_placement=True))


def clean_df_y(df):
    '''
    Input:
    df = clean dataframe

    Creates bins for all differences. All bins labels are roughly calculated
    based on winning probability from original ELO equation:

    1/(1+10^m), where m = (elo difference)/400
    Also, bins the start time
    Returns:

    df = cleaner dataframe
    y = results as a 1D numpy array
    '''

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
    X = array used for prediction
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
    '''
    Input:
    df = cleaned dataframe
    y = all result values in an Numpy Array
    splt = Split size for test set in % as 0.80 or # as 200
    cols = list of columns to create X values to predict over

    This function creates X array, X_train, X_test, y_train, and y_test.
    If the columns are not elo difference or color, it creates dummy columns.

    Returns:
    X = values to run predictions 
    X_train = training prediction set
    X_test = testing prediction set
    y_train = training result set
    y_test = testing result set
    '''

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
        elif (cols[0] == 'diff_bin' or cols[0] == 'color') and cols[1] != 'color':
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
        elif (cols[0] == 'diff_bin' or cols[0] == 'color') and cols[1] != 'color':
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


def gc_classifier(optm, lss):
    '''
    Input:
    optm = Optimizer
    lss = Loss Function

    Creates Keras Sequential Model.
    input layer: 64 units, softmax activation
    hidden layer # 1: 128 units, relu activation
    hidden layer # 2: 32 units, softmax activation
    output layer: sigmoid activation
    metrics: 'accuracy'

    Returns:
    classifier = Created model
    '''

    classifier = Sequential()
    classifier.add(Dense(units=64, activation='softmax', input_dim=X.shape[1]))
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=32, activation='softmax'))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(
        optimizer=optm, loss=lss, metrics=['accuracy'])

    return classifier


def run_classifier():

    df = pd.read_csv('../data/use_for_predictions.csv')
    df = df.loc[df['result'] != 0.5].copy()
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)

    df, y = clean_df_y(df)
    X_train, X_test, y_train, y_test, X = xy_custom(
        df, y, 100, ['diff_bin', 'time_bin', 'game_time', 'weekday'])
    std_sclr = StandardScaler()
    X_train = std_sclr.fit_transform(X_train)
    X_test = std_sclr.fit_transform(X_test)

    classifier = KerasClassifier(build_fn=gc_classifier)

    parameters = {'batch_size': [8, 20, 44, 92], 'nb_epoch': [64, 128, 256],
                  'optm': ['nadam', 'adagrad', 'rmsprop', 'adam', 'adadelta'],
                  'lss': ['mae', 'mse', 'binary_crossentropy']}

    grid_search = GridSearchCV(classifier, parameters, scoring='accuracy', cv=11,
                               return_train_score=True)

    grid_search.fit(X=X_train, y=y_train, verbose=1)

    return grid_search, classifier

grid_search, classifier_ = run_classifier()
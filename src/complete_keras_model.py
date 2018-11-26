# Author: Mohit Gangwani
# Date: 11/21/2018
# Git-Hub: Data-is-Life

import pandas as pd
import numpy as np

from pandas import get_dummies as gd

from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from time import time


df = pd.read_csv('../data/use_for_predictions.csv')
df = df.loc[df['result'] != 0.5].copy()
df.reset_index(inplace=True)
df.drop(columns=['index'], inplace=True)


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


df, y = clean_df_y(df)


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


def _classifier():

    classifier = Sequential(optm, lss)
    classifier.add(Dense(units=64, activation='softmax', input_dim=X.shape[1]))
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=32, activation='softmax'))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(
        optimizer=optm, loss=lss, metrics=['accuracy'])

    return classifier


all_cols = [
    ['diff_bin', 'color', 'time_bin', 'game_time', 'weekday'],
    ['diff_bin', 'time_bin', 'game_time', 'weekday'],
    ['diff_bin', 'color', 'time_bin', 'weekday'],
    ['diff_bin', 'time_bin', 'weekday'],
    ['diff_bin', 'color', 'time_bin', 'game_time'],
    ['diff_bin', 'time_bin', 'game_time'],
    ['diff_bin', 'color', 'time_bin'],
    ['diff_bin', 'time_bin'],
    ['diff_bin', 'color', 'game_time', 'weekday'],
    ['diff_bin', 'game_time', 'weekday'],
    ['diff_bin', 'color', 'weekday'],
    ['diff_bin', 'weekday'],
    ['diff_bin', 'color', 'game_time'],
    ['diff_bin', 'game_time'],
    ['diff_bin', 'color'],
    ['diff_bin']]


results = {}

lossess = ['mae', 'binary_crossentropy']
optimiz = ['nadam', 'rmsprop', 'adagrad', 'adam']

for ls in lossess:
    for opm in optimiz:
        for clm in all_cols:
            st = time()
            X_train, X_test, y_train, y_test, X = xy_custom(df, y, 100, clm)
            std_sclr = StandardScaler()
            X_train = std_sclr.fit_transform(X_train)
            X_test = std_sclr.fit_transform(X_test)
            for bs in [8, 20, 44, 92]:
                print(bs)
                for ep in [50, 100, 200]:
                    classifier = _classifier(ls, opm)
                    classifier.fit(X_train, y_train, batch_size=bs, epochs=ep,
                                   class_weight='balanced', shuffle=False,
                                   verbose=2)
                    y_pred = classifier.predict(X_test)
                    y_pred = (y_pred > 0.5)
                    cm = confusion_matrix(y_test, y_pred)
                    results[(f'c{clm}-b{bs}-e{ep}-l{ls}-o{opm}')] = [
                        cm, (f'{((cm[0][0]+cm[1][1])/cm.sum()*100).round(1)}%')]
        print(clm)
        print(opm)
        print(ls)
        print(results)
        print(time() - st)


'''
    The data containg 2,127 total games played. To devise a proper way to 
    rectify the game probability to be only dependant on Difference in ELO,
    following pre-game parameters were used:
    1. Difference in ELO
    2. Assigned Color
    3. Time allowed per player (aka game time)
    4. Time of the Day
    5. Day of the Week

    To properly assess which of these factors played the biggest impact, the
    decision to try different combinations was adopted. Here are all the
    combinations possible, represented as thier number on the list above:
    - Single parameters only (1, 2, 3, 4, 5)
    - Combination of two parameters (1+2, 1+3, 1+4, 1+5, 2+3, 2+4, 2+5, 3+4,
      3+5, 4+5)
    - Combination of three parameters (1+2+3, 1+2+4, 1+2+5, 1+3+4, 1+3+5, 1+4+5,
      2+3+4, 2+3+5, 2+4+5, 3+4+5)
    - Combination of four parameters (1+2+3+4, 1+2+3+5, 1+2+4+5, 1+3+4+5, 2+3+4+5)
    - Combination of all five parameters (1+2+3+4+5)
    
    This ended up to be 31 different combinations. Ideally any of them could be
    the best estimators. After running only one test, all the parameters devoid
    of Difference in ELO were dropped, since the accuracy in any of test was
    not even close to 66%.

    That left 16 total parameters to run tests on:
    - Single parameters only (1)
    - Combination of two parameters (1+2, 1+3, 1+4, 1+5)
    - Combination of three parameters (1+2+3, 1+2+4, 1+2+5, 1+3+4, 1+3+5, 1+4+5)
    - Combination of four parameters (1+2+3+4, 1+2+3+5, 1+2+4+5, 1+3+4+5)
    - Combination of all five parameters (1+2+3+4+5)

    With a small subset of games data, train test split was made based on
    personal experience. The test set was 100 games and the remainder were
    used for training (2027 games). This is a roughly 95/5 split.

    To make the prediction model perform better, difference in ELO and the
    time of the day were binned.

    The time of the day was easy to bin, which was to bin it in to each hour
    of the day (Total of 24 bins.)

    To properly bin the difference in ELO, a different aproach was used. The
    bins were created in the following manner:
    - Intervals of 100 for difference in ELO being above 600 or below -600
    - Intervals of 50 for difference in ELO being between 400 to 600 and 
      between -600 and -300
    - Intervals of 25 for difference in ELO being between 250 to 325 and 
      between -250 and -200
    - Intervals of 10 for difference in ELO being between 110 to 200 and 
      between -200 and -100
    - Intervals of 5 for difference in ELO being between -100 to 100

    All bins labels were calculated and assigned roughly based on winning
    probability using original chances of winning ELO equation:
    1/(1+10^m), where m = (elo difference)/400

    Also, dummy columns were created for:
    - Time of the day
    - Game time
    - Day of the week
    
    The Keras Sequential Model was used to determine which combination of these
    parameters gave the best results.

    The Keras Classifier created has the following layers:
    - Input layer:
        - Activation Function = SoftMax
        - Units = 64
    - First hidden Layer:
        - Activation Function = ReLu
        - Units = 128
    - Second (and final) hidden layer:
        - Activation Function = SoftMax
        - Units = 32
    - Output layer:
        - Activation Function = Sigmoid

    To compile the classifier, 'accuracy' (aka 'binary_accuracy') was
    determined for the to be the best option for the metrics to determine the
    validity of the model, since it has to determine only between win or lose.

    Idea of Grid Search was rejected, primarilly because of the amount of data
    available. Since the training set is of only 2,027 games, the grid-search
    will be using 10% to 20% of it as validation data, and being a fairly
    beginner at chess, the playing style is ever so evolving, any part of the
    data ommitted will lead to picking the wrong parameters.

    Hence, to get a better idea of which parameters will work the best,
    "For-Loop" is used to train the classifier with different parameters.
    These following parameters are used in that for loop:

    1. Losses:
        1. Mean Absolute Error (Measure of difference between two continuous
           variables)
        2. Binary Crossentropy (AKA Minimizing Log Loss)
        3. Meas Square Error (Mean squared difference between the estimated
           values and what is estimated)
    2. Optimizers:
        1. Nadam (Adam RMSprop with Nesterov momentum)
        2. RMSProp (Divides the gradient by running mean of recent magnitude)
        3. AdaGrad (optimizer with parameter-specific LR, adapted relative to
           how frequently a parameter gets updated during training)
        4. Adam (ADAptive Moment estimation)
        5. AdaDelta (Adapts LR based on a moving window of gradient updates)
    3. Batch Sizes:
        1. 8
        2. 20
        3. 44
        4. 92
    4. Epochs:
        1. 50
        2. 100
        3. 200

    Please note, that SGD, Squared Hinge, and Hinge were also tried, but they
    all failed to give any resonable predictions. All of them predicted all
    wins, hence were dropped after running test on a single batch.

    Currently all tests are being run on different machines (Kaggle Kernel,
    personal computer, and AWS EC2.)

    The results will be published in the "docs" folder once the ideal
    combinations have been determined.


'''
# Author: Mohit Gangwani
# Date: 11/05/2018
# Git-Hub: Data-is-Life

import pandas as pd
import numpy as np

from pandas import get_dummies as gd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


def get_Xy_train_test(df, s_min=0.8, s_max=0.85):
    '''
    df = dataframe for prediction
    s_min = percentage for minimum split %. Default value = 0.8
    s_max = percentage for highest split %. Default value = 0.85

    Drops games that were drawn
    Bins difference in elo
    creates dummies
    returns:
    X_train = Training set with all features
    y_train = Training set with results
    X_test = Testing set with all features
    y_test = Testing set with results to compare with predictions
    df = dataframe with all columns binned and dummy columns created
    '''

    df = df.loc[df['result'] != 0.5].copy()

    # df = df.loc[1000:]

    y = np.array(df['result'])
    

    dif_bins = list(range(-1000, -399, 100))
    dif_bins.extend(list(range(-390, -199, 10)))
    dif_bins.extend(list(range(-195, 196, 5)))
    dif_bins.extend(list(range(200, 401, 10)))
    dif_bins.extend(list(range(500, 1001, 100)))

    dif_lbl = list(range(len(dif_bins) - 1))

    df.loc[:, 'diff_bin'] = pd.cut(df['diff'], bins=dif_bins, labels=dif_lbl)

    df.loc[:, 'time_bin'] = pd.cut(df['start_time'], bins=24, labels=False)

    df.drop(columns=['result', 'opp_elo', 'elo', 'start_time', 'diff', 'day'],
            inplace=True)

    df = gd(df, prefix='gt', drop_first=True, columns=['game_time'])
    df = gd(df, prefix='st', drop_first=True, columns=['start_time_bin'])
    df = gd(df, prefix='wd', drop_first=True, columns=['weekday'])

    X = df.values    

    rand_split = np.random.randint(int(len(X) * s_min), int(len(X) * s_max))

    X_train = X[:rand_split]
    y_train = y[:rand_split]
    X_test = X[rand_split:]
    y_test = y[rand_split:]

    print(f'X Shape: {X.shape}')
    print(f'y Shape: {y.shape}')
    print(f'X_train Shape: {X_train.shape}')
    print(f'X_test Shape: {X_test.shape}')
    print(f'y_train Shape: {y_train.shape}')
    print(f'y_test Shape: {y_test.shape}')

    return X_train, X_test, y_train, y_test, X, y, df


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

# df = pd.read_csv('../data/use_for_predictions.csv')
# df = df.loc[df['result'] != 0.5].copy()
# df.reset_index(inplace=True)
# df.drop(columns=['index'], inplace=True)

# df, y = clean_df_y(df)


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


def cross_val_process(classifier, X_train, y_train, X_test, y_test, cv,
                      scoring='average_precision'):
    '''
    Classifer = Classifier created
    X_train = Training set with all features
    y_train = Training set with results
    X_test = Testing set with all features
    y_test = Testing set with results to compare with predictions
    cv = number of tests to run
    Scoring = type of scoring. Default value = "average_precision"

    Runs cross validation(CV) on the given model with Training set and prints
    the following analysis:
    - Average Accuracy for scores from CV
    - Standard Deviation for scores from CV
    - Scores from CV
    - Feature importance (if available)

    Also, creates:
    - Prediction values from X_test
    - confusion matrix comparing predicted values to y_test

    Other scorings available:

    For Classification:
       "accuracy", "balanced_accuracy", "average_precision", "brier_score_loss",
       "f1", "f1_micro", "f1_macro", "f1_weighted", "f1_samples", "neg_log_loss",
       "precision", "recall", "roc_auc"

    For Clustering:
       "adjusted_mutual_info_score", "adjusted_rand_score", "completeness_score",
       "fowlkes_mallows_score", "homogeneity_score", "mutual_info_score",
       "normalized_mutual_info_score", "v_measure_score"

    For Regression:
       "explained_variance", "neg_mean_absolute_error", "neg_mean_squared_error",
       "neg_mean_squared_log_error", "neg_median_absolute_error", "r2"
    '''

    cross_val = cross_val_score(classifier, X_train, y_train, cv=cv,
                                scoring=scoring)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    pred_acc = round(classifier.score(X_test, y_test) * 100, 2)

    print(f'Average_Accuracy({scoring})={round(cross_val.mean() * 100, 2)}%')
    print(f'Scores({scoring})={cross_val.round(3)}')
    print(f'Standard_Deviation={round(cross_val.std(), 3)}\n')

    print(f'Prediction_Confusion_Matrix:\n[{cm[0][0]}|{cm[0][1]}]\
    	                                \n[{cm[1][0]}|{cm[1][1]}]')
    print(f'Prediction_Accuracy={pred_acc}%')

    try:
        print(f'Feature importance = {classifier.feature_importances_}')
    except:
        print("No Feature Importances")

    print(
        f'\nClassification Report:\n{classification_report(y_test, y_pred)}\n')

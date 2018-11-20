# Author: Mohit Gangwani
# Date: 11/05/2018
# Git-Hub: Data-is-Life

import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from math import ceil


def get_Xy_train_test(df, split_min=0.8, split_max=0.85):
    '''Classifer = Classifier created
    df = dataframe for prediction
    split_min = percentage for minimum split %. Default value = 0.8
    * New: If need an exact number of games to predict, pass an integer

    split_max = percentage for highest split %. Default value = 0.85

    Drops games that were drawn
    Bins difference in elo
    creates dummies
    returns:
    X_train, X_test, y_train, y_test
    '''

    df = df[df['result'] != 0.5]

    # df = df.loc[1000:]

    y = np.array(df['result'])
    print(f'y Shape: {y.shape}')

    dif_bins = list(range(-1000, -399, 100))
    dif_bins.extend(list(range(-390, -199, 10)))
    dif_bins.extend(list(range(-195, 196, 5)))
    dif_bins.extend(list(range(200, 401, 10)))
    dif_bins.extend(list(range(500, 1001, 100)))

    dif_labels = list(range(len(dif_bins) - 1))

    df.loc[:, 'diff_bin'] = pd.cut(
        df['diff'], bins=dif_bins, labels=dif_labels)

    df.loc[:, 'start_time_bin'] = pd.cut(
        df['start_time'], bins=24, labels=False)

    df.drop(columns=['Unnamed: 0', 'result', 'opp_elo', 'elo', 'start_time',
                     'diff', 'day'], inplace=True)
    print('.')

    # df = pd.get_dummies(df, prefix='gt', drop_first=True,
    #                     columns=['game_time'])
    df = pd.get_dummies(df, prefix='st', drop_first=True,
                        columns=['start_time_bin'])
    df = pd.get_dummies(df, prefix='wd', drop_first=True, columns=['weekday'])

    # df.drop(columns=['start_time', 'game_time', 'diff', ])

    X = df.values
    print(f'X Shape: {X.shape}')

    rand_split = np.random.randint(
            int(len(X) * split_min), int(len(X) * split_max))

    X_train = X[:rand_split]
    y_train = y[:rand_split]
    X_test = X[rand_split:]
    y_test = y[rand_split:]

    print(f'X_train Shape: {X_train.shape}')
    print(f'X_test Shape: {X_test.shape}')
    print(f'y_train Shape: {y_train.shape}')
    print(f'y_test Shape: {y_test.shape}')

    return X_train, X_test, y_train, y_test, X, y, df


def cross_validation_process(classifier, X_test, y_test, cv=5,
                             scoring='average_precision'):
    '''Classifer = Classifier created
       X_test = X_test
       y_test = y_test
       cv = number of tests to run. Default value = 5
       Scoring = type of scoring. Default value = "average_precision"

       Runs cross validation(CV) on a model and test on test set. Prints:
       - Average Accuracy for scores from CV
       - Standard Deviation for scores from CV
       - Scores from CV
       - Feature importance (if available)
       - Confusion matrix for test set
       - Accuracy % for test set

       Returns Scores from CV

       Other scorings:

       Classification:
       accuracy, balanced_accuracy, average_precision, brier_score_loss, f1,
       f1_micro, f1_macro, f1_weighted, f1_samples, neg_log_loss, precision,
       recall, roc_auc

       Clustering:
       adjusted_mutual_info_score, adjusted_rand_score, completeness_score,
       fowlkes_mallows_score, homogeneity_score, mutual_info_score,
       normalized_mutual_info_score, v_measure_score

       Regression:
       explained_variance, neg_mean_absolute_error, neg_mean_squared_error,
       neg_mean_squared_log_error, neg_median_absolute_error, r2'''

    scores = cross_val_score(classifier, X_test, y_test, cv=cv,
                             scoring=scoring)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    pred_acc = round(classifier.score(X_test, y_test) * 100, 2)

    print(f'Average_Accuracy({scoring})={round(scores.mean()*100,2)}%')
    print(f'Standard_Deviation={round(scores.std(),3)}')
    print(f'True_Score(Mean/SD)={round(scores.mean()/scores.std(),3)}\n')
    print(f'Prediction_Confusion_Matrix=[{cm[0][0]}|{cm[0][1]}]:[{cm[1][0]}|{cm[1][1]}]')
    print(f'Prediction_Accuracy={pred_acc}%')
    # print(f'Scores({scoring})={scores.round(3)}')
    # try:
    #     print(f'Feature importance = {classifier.feature_importances_}')
    # except:
    #     print("No Feature Importances")
    # print(
    # f'\nClassification Report:\n{classification_report(y_test, y_pred)}\n')

    # print('-----------------------------------------------------------------------')
    return scores

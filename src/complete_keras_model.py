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

from dummies_bins_test_train_cv import clean_df_y
from dummies_bins_test_train_cv import xy_tt
from dummies_bins_test_train_cv import xy_custom


def ann_classifier(optm, lss):
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

    classifier = Sequential(optm, lss)
    classifier.add(Dense(units=64, activation='softmax', input_dim=X.shape[1]))
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=32, activation='softmax'))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(
        optimizer=optm, loss=lss, metrics=['accuracy'])

    return classifier

# They are sorted decsending by the number of columns
# all_cols = [
#     ['diff_bin', 'color', 'time_bin', 'game_time', 'weekday'],
#     ['diff_bin', 'time_bin', 'game_time', 'weekday'],
#     ['diff_bin', 'color', 'time_bin', 'weekday'],
#     ['diff_bin', 'time_bin', 'weekday'],
#     ['diff_bin', 'color', 'time_bin', 'game_time'],
#     ['diff_bin', 'time_bin', 'game_time'],
#     ['diff_bin', 'color', 'time_bin'],
#     ['diff_bin', 'time_bin'],
#     ['diff_bin', 'color', 'game_time', 'weekday'],
#     ['diff_bin', 'game_time', 'weekday'],
#     ['diff_bin', 'color', 'weekday'],
#     ['diff_bin', 'weekday'],
#     ['diff_bin', 'color', 'game_time'],
#     ['diff_bin', 'game_time'],
#     ['diff_bin', 'color'],
#     ['diff_bin']]

# # Creating an empty dictionary to store all results
# results = {}

# # Three losses cosidered to iterate over
# lossess = ['mae', 'binary_crossentropy', 'mse']
# # Five optimizers to iterate over
# optimiz = ['nadam', 'rmsprop', 'adagrad', 'adam', 'adadelta']
# # Four batch sizes to test the model
# b_s = [8, 20, 44, 92]
# # Three epochs to run all batches
# e_p = [50, 100, 200]


# for ls in lossess:
#     for opm in optimiz:
#         for clm in all_cols:
#             st = time()
#             X_train, X_test, y_train, y_test, X = xy_custom(df, y, 100, clm)
#             std_sclr = StandardScaler()
#             X_train = std_sclr.fit_transform(X_train)
#             X_test = std_sclr.fit_transform(X_test)
#             for bs in b_s:
#                 print(bs)
#                 for ep in e_p:
#                     classifier = ann_classifier(ls, opm)
#                     classifier.fit(X_train, y_train, batch_size=bs, epochs=ep,
#                                    class_weight='balanced', shuffle=False,
#                                    verbose=2)
#                     y_pred = classifier.predict(X_test)
#                     y_pred = (y_pred > 0.5)
#                     cm = confusion_matrix(y_test, y_pred)
#                     results[(f'c{clm}-b{bs}-e{ep}-l{ls}-o{opm}')] = [
#                         cm, (f'{((cm[0][0]+cm[1][1])/cm.sum()*100).round(1)}%')]
#         print(clm)
#         print(opm)
#         print(ls)
#         print(results)
#         print(time() - st)

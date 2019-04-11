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

from dummies_bins_test_train_cv import initial_df
from dummies_bins_test_train_cv import bin_df_get_y
from dummies_bins_test_train_cv import partial_df
from dummies_bins_test_train_cv import xy_custom


def ann_classifier(X):
    '''
    Input:
    X = X values

    Creates Keras Sequential Model.
    input layer: 128 units, ReLu activation
    hidden layer # 1: 64 units, ReLu activation
    output layer: sigmoid activation
    metrics: 'accuracy'
    loss: Binary Crossentropy

    Returns:
    classifier = Created model
    '''

    classifier = Sequential()

    classifier.add(Dense(units=128, kernel_initializer='uniform',
                   activation=K.relu, input_dim=X.shape[1]))
    classifier.add(Dense(units=64, kernel_initializer='uniform',
                   activation=K.relu))
    classifier.add(Dense(units=1, kernel_initializer='uniform',
                   activation=K.sigmoid))
    
    classifier.compile(optimizer='rmsprop', loss=K.binary_crossentropy,
                       metrics=['accuracy'])

    return classifier

# They are sorted decsending by the number of columns
# all_cols = [
    # ['diff_bin'],
    # ['diff_bin', 'game_time'],
    # ['diff_bin', 'color'],
    # ['diff_bin', 'color', 'game_time'],
    # ['diff_bin', 'time_bin'],
    # ['diff_bin', 'game_time', 'time_bin'],
    # ['diff_bin', 'color', 'time_bin'],
    # ['diff_bin', 'color', 'game_time', 'time_bin'],
    # ['diff_bin', 'time_bin', 'weekday'],
    # ['diff_bin', 'game_time', 'time_bin', 'weekday'],
    # ['diff_bin', 'color', 'time_bin', 'weekday'],
    # ['diff_bin', 'color', 'game_time', 'time_bin', 'weekday']]

# # Creating an empty dictionary to store all results
# bs_ep_dict = {5: [80, 160], 10: [150, 300], 15: [200, 400]}
# for splt in [1/11, 1/15, 1/20]:
# bs_ep_dict = {6: [72, 144], 12: [150, 300], 30: [360, 720]}
# results = {}

# df, df_len = initial_df('../input/use_for_predictions.csv')

# for splt in [1/6]:
#     df_s = partial_df(df, splt)
#     df_s, y = bin_df_get_y(df_s)
#     for cv in range(11):
#         st = time()
#         for bs, epov in bs_ep_dict.items():
#             for ep in epov:
#                 X_train, X_test, y_train, y_test, X = xy_custom(
#                     df_s, y, 20, ['diff_bin', 'game_time'])
#                 ann_model = ann_classifier(X_test)
#                 ann_model.fit(X_train, y_train, batch_size=bs, epochs=ep,
#                               class_weight='balanced', shuffle=False, verbose=3)
#                 y_pred = ann_model.predict(X_test)
#                 y_pred = (y_pred >= 0.5)
#                 cm = confusion_matrix(y_test, y_pred)
#                 results[(f'b{bs}-e{ep}-{splt}-{cv}-{time()-st}')] = [
#                     cm, (f'{((cm[0][0]+cm[1][1])/cm.sum()*100).round(2)}%')]
#             print(results)

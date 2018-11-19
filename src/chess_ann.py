#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 17:38:34 2018

@author: guess
"""

import pandas as pd
# import numpy as np

from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras import regularizers

from dummies_bins_test_train_cv import get_Xy_train_test
# from clean_chess_game_log import main_cleanup

# _, _, _ = main_cleanup('../data/dest.pgn')
df = pd.read_csv('../data/use_for_predictions.csv')

X_train, X_test, y_train, y_test, X, y, df_clean = get_Xy_train_test(df, .97, .975)
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
y_train = y_train.astype('int64')
y_test = y_test.astype('int64')

std_sclr = StandardScaler()
X_train = std_sclr.fit_transform(X_train)
X_test = std_sclr.fit_transform(X_test)

for num in [10, 15, 30, 50, 75, 100, 150, 200]:

    def _classifier():
        classifier = Sequential()

        classifier.add(Dense(units=64, kernel_initializer='uniform',
                             activation='relu', input_dim=30))

        classifier.add(
            Dense(units=256, kernel_initializer='uniform',
                  activation='softmax'))

    #    classifier.add(
    #        Dense(units=256, kernel_initializer='uniform',
    # activation='softplus'))
    #
    #    classifier.add(Dropout(rate=0.15))
    #
        classifier.add(
            Dense(units=128, kernel_initializer='uniform', activation='relu'))

        classifier.add(
            Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

        classifier.compile(
            optimizer='nadam', loss='binary_crossentropy',
            metrics=['accuracy'])

        return classifier

    classifier = _classifier()

    classifier.fit(X_train, y_train, batch_size=8, epochs=num,
                   class_weight='balanced', shuffle=False)

    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

    cm = confusion_matrix(y_test, y_pred)

    print(f'{((cm[0][0]+cm[1][1])/cm.sum()*100).round(1)}%')

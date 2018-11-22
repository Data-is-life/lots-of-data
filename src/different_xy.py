# Author: Mohit Gangwani
# Date: 11/21/2018
# Git-Hub: Data-is-Life

import pandas as pd
import numpy as np


def diff_bins(df):

    df = df.loc[df['result'] != 0.5].copy()
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)

    dif_bn = list((range(-1000, -499, 100)))
    dif_bn.extend(list((range(-400, -149, 50))))
    dif_bn.extend(list((range(-100, 101, 10))))
    dif_bn.extend(list((range(150, 401, 50))))
    dif_bn.extend(list((range(500, 1001, 100))))

    dif_lbl = list(range(1, 5))
    dif_lbl.extend(list(range(6, 37, 3)))
    dif_lbl.extend(list(range(38, 63, 2)))
    dif_lbl.extend(list(range(65, 96, 3)))
    dif_lbl.extend(list(range(96, 101)))

    df.loc[:, 'diff_bin'] = pd.cut(df['diff'], bins=dif_bn, labels=dif_lbl)
    df.loc[:, 'time_bin'] = pd.cut(df['start_time'], bins=24, labels=False)

    y = np.array(df['result'])

    df.drop(columns=['result', 'opp_elo', 'elo', 'start_time', 'diff', 'day'],
            inplace=True)

    return df, y


def xy_tt(X, y, s_mn=0.8, s_mx=0.85):

    rand_split = np.random.randint(int(len(X) * s_mn), int(len(X) * s_mx))

    X_train = X[:rand_split]
    y_train = y[:rand_split]
    X_test = X[rand_split:]
    y_test = y[rand_split:]

    print(f'y Shape: {y.shape}')
    print(f'X Shape: {X.shape}')
    print(f'X_train Shape: {X_train.shape}')
    print(f'X_test Shape: {X_test.shape}')
    print(f'y_train Shape: {y_train.shape}')
    print(f'y_test Shape: {y_test.shape}')

    return X_train, X_test, y_train, y_test


def xy_custom(df, s_mn, s_mx, cols):

    df, y = diff_bins(df)

    if cols == ['diff_bin']:
        df = df[['diff_bin']]
        X = df.values
        X = X.reshape(-1, 1)

    else:
        df = df[cols].copy()

        if cols == ['diff_bin', 'color']:
            X = df.values

        else:
            df = pd.get_dummies(df, prefix='gt', drop_first=True,
                                columns=['game_time'])

            if cols == ['diff_bin', 'color', 'game_time']:
                X = df.values

            else:
                df = pd.get_dummies(df, prefix='st', drop_first=True,
                                    columns=['time_bin'])

                if cols == ['diff_bin', 'color', 'game_time', 'start_time']:
                    X = df.values

                else:
                    df = pd.get_dummies(df, prefix='wd', drop_first=True,
                                        columns=['weekday'])
                    X = df.values

    X_train, X_test, y_train, y_test = xy_tt(X, y, s_mn, s_mx)

    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')
    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')

    return X_train, X_test, y_train, y_test, X, y



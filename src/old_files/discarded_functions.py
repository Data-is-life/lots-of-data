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
    PRE = 'abcdefghijklmnopqrstuvwxyz'

    if len(cols) == 1:
        X = df_n.values
        X = X.reshape(-1, 1)

    elif len(cols) == 2:
        if cols[1] == 'color':
            X = df_n.values
        else:
          df_n = gd(df_n, prefix='a', drop_first=True, columns=[cols[1]])
          X = df_n.values

    elif len(cols) == 3:
        if cols[1] == 'color':
            df_n = gd(df_n, prefix='a', drop_first=True, columns=[cols[2]])
            X = df_n.values
        else:
            df_n = gd(df_n, prefix='a', drop_first=True, columns=[cols[1]])
            df_n = gd(df_n, prefix='b', drop_first=True, columns=[cols[2]])
            X = df_n.values

    elif len(cols) == 4:
        if cols[1] == 'color':
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

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')

    return X_train, X_test, y_train, y_test, X

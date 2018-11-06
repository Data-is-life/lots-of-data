import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score


def get_Xy_train_test(df):
    
    df = df[df['result'] != 0.5]

    y = np.array(df['result'])
    print(f'y Shape: {y.shape}')

    bins = list(range(-2000, -499, 50))
    bins.extend(list(range(-490, -199, 10)))
    bins.extend(list(range(-195, 196, 5)))
    bins.extend(list(range(200, 501, 10)))
    bins.extend(list(range(550, 2001, 50)))

    labels = list(range(len(bins) - 1))

    df['diff_bin'] = pd.cut(df['diff'], bins=bins, labels=labels)

    numeric_predictors = ['color', 'diff_bin',
                          'game_time', 'start_time', 'weekday']
    df = df[numeric_predictors]

    df = pd.get_dummies(df, prefix='gt', drop_first=True,
                        columns=['game_time'])
    df = pd.get_dummies(df, prefix='st', drop_first=True,
                        columns=['start_time'])
    df = pd.get_dummies(df, prefix='wd', drop_first=True, columns=['weekday'])

    X = df.values
    print(f'X Shape: {X.shape}')

    rand_split = np.random.randint(int(len(X) * .925), int(len(X) * .975))

    X_train = X[:rand_split]
    y_train = y[:rand_split]
    X_test = X[rand_split:]
    y_test = y[rand_split:]

    print(f'X_train Shape: {X_train.shape}')
    print(f'X_test Shape: {X_test.shape}')
    print(f'y_train Shape: {y_train.shape}')
    print(f'y_test Shape: {y_test.shape}')

    return X_train, X_test, y_train, y_test


def cross_validation_process(classifier, X_test, y_test, cv=5, scoring='average_precision'):
    '''Classifer = Classifier created
       X_test = X_test
       y_test = y_test
       cv = number of tests to run. Default value = 5
       Scoring = type of scoring. Default value = "average_precision"

       Other scorings:

       Classification:
       accuracy, balanced_accuracy, average_precision, brier_score_loss, f1, f1_micro,
       f1_macro, f1_weighted, f1_samples, neg_log_loss, precision, recall, roc_auc

       Clustering:
       adjusted_mutual_info_score, adjusted_rand_score, completeness_score,
       fowlkes_mallows_score, homogeneity_score, mutual_info_score,
       normalized_mutual_info_score, v_measure_score

       Regression:
       explained_variance, neg_mean_absolute_error, neg_mean_squared_error, 
       neg_mean_squared_log_error, neg_median_absolute_error, r2'''

    scores = cross_val_score(
        classifier, X_test, y_test, cv=cv, scoring=scoring)
    print(f'Average Accuracy({scoring}) = {round(scores.mean()*100,2)}%')
    print(f'\nStandard Deviation = {round(scores.std(),3)}')
    print(f'\nScores({scoring}):\n {scores}')
    y_pred = classifier.predict(X_test)
    print(
        f'\nClassification Report:\n{classification_report(y_test, y_pred)}\n')
    cm = confusion_matrix(y_test, y_pred)
    print(f'confusion_matrix:\n{cm}\n')
    print(f'confusion_matrix accuracy: {(cm[0][0]+cm[1][1])/cm.sum()*100}%')
    print(f'-----------------------------------------------------------------------')
    return scores
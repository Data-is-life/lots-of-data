
from all_classification_models import run_classifier_model
from dummies_bins_test_train_cv import initial_df
from col_info import all_cols
from time import time, sleep
from numpy import arange
import warnings
import gc
gc.enable()
warnings.filterwarnings('ignore')

df = initial_df('../data/use_for_predictions.csv')


def run_lda_qda_ridge(df):
    sleep(60*8*60)
    st = time()
    run_classifier_model(df, 25, all_cols, 'lda', 0, 0)
    print(f'Finished LDA in {time()-st} Seconds')
    gc.collect()

    for j in [0.1, 0.5, 1.0]:
        print(f'QDA-{j}')
        st = time()
        run_classifier_model(df, 25, all_cols, 'qda', j, 0)
        print(f'Finished QDA #{j} in {time()-st} Seconds')
        gc.collect()

    for j in arange(-1, 4.01, 0.5):
        print(f'Ridge-{j}')
        st = time()
        run_classifier_model(df, 25, all_cols, 'ridge_bal', j, 0)
        run_classifier_model(df, 25, all_cols, 'ridge_unbal', j, 0)
        print(f'Finished Ridge Bal & Unbal #{j} in {time()-st} Seconds\n')
        gc.collect()


def run_logi_knn(df):
    for j in range(5, 40):
        print(f'KNN-{j}')
        st = time()
        run_classifier_model(df, 25, all_cols, 'knn', j=j, z=0)
        print(f'Finished KNN #{j} in {time()-st} Seconds')
        gc.collect()

    for j in arange(-2, 3.01, 0.25):
        print(f'Logistic-{j}')
        st = time()
        run_classifier_model(df, 25, all_cols, 'logistic_bal', j, 0)
        run_classifier_model(df, 25, all_cols, 'logistic_unbal', j, 0)
        print(f'Finished Logistic Bal & Unbal #{j} in {time()-st} Seconds')
        gc.collect()


def run_rf_nine_twelve(df):
    for j in range(10, 13):
        print(f'Random Forest-{j}')
        st = time()
        run_classifier_model(df, 25, all_cols, 'random_forest_bal', j, 0)
        run_classifier_model(df, 25, all_cols, 'random_forest_unbal', j, 0)
        print(f'Finished Random Forest #{j} in {time()-st} Seconds')
        gc.collect()


def run_rf_one_nine(df):
    for j in range(1, 10):
        print(f'Random Forest-{j}')
        st = time()
        run_classifier_model(df, 25, all_cols, 'random_forest_bal', j, 0)
        run_classifier_model(df, 25, all_cols, 'random_forest_unbal', j, 0)
        print(f'Finished Random Forest #{j} in {time()-st} Seconds')
        gc.collect()


def run_svc_abc(df):
    for j in range(1, 10):
        print(f'AdaBoost-{j}')
        st = time()
        run_classifier_model(df, 25, all_cols, 'abc', j, 0)
        print(f'Finished AdaBoost #{j} in {time()-st} Seconds')
        gc.collect()
    for j in range(3, 11, 2):
        print(f'SVC-{j}')
        st = time()
        run_classifier_model(df, 25, all_cols, 'svc', j, 0)
        print(f'Finished SVC #{j} in {time()-st} Seconds')
        gc.collect()


#run_svc_abc(df)
#run_lda_qda_ridge(df)
#run_logi_knn(df)
#run_rf_nine_twelve(df)
#run_rf_one_nine(df)

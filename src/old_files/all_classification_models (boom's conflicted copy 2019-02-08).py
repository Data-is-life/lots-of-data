import gc
from pandas import DataFrame

from dummies_bins_test_train_cv import partial_df
from dummies_bins_test_train_cv import xy_custom
from dummies_bins_test_train_cv import bin_df_get_y

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RidgeClassifier as RdC
from sklearn.neighbors import KNeighborsClassifier as KNNc
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix as CMx
from sklearn.metrics import roc_auc_score as RAS
from sklearn.metrics import accuracy_score as ASc
from sklearn.metrics import balanced_accuracy_score as BAs
from sklearn.metrics import f1_score as F1S
from sklearn.metrics import brier_score_loss as BSL
from sklearn.metrics import log_loss as LL
import datetime
from time import sleep

gc.enable()


def get_classifier(classifier, df_s, j=0, z=0):
    if classifier.upper() == 'LDA':
        CLSFR = LDA(solver='lsqr', shrinkage='auto')

    elif classifier.lower() == 'logistic_bal':
        CLSFR = LR(class_weight='balanced', random_state=5, max_iter=1e4,
                   C=0.1**j, solver='newton-cg')

    elif classifier.lower() == 'logistic_unbal':
        CLSFR = LR(random_state=5, max_iter=1e4, C=0.1**j, solver='newton-cg')

    elif classifier.upper() == 'KNN':
        CLSFR = KNNc(n_neighbors=j)

    elif classifier.lower() == 'ridge_bal':
        CLSFR = RdC(alpha=j, class_weight='balanced', random_state=5)

    elif classifier.lower() == 'ridge_unbal':
        CLSFR = RdC(alpha=j, random_state=5)

    elif classifier.lower() == 'random_forest_bal':
        CLSFR = RFC(n_estimators=int(50*j), random_state=5, 
                    min_samples_leaf=2, class_weight='balanced')

    elif classifier.lower() == 'random_forest_unbal':
        CLSFR = RFC(n_estimators=int(50*j), random_state=5,
                    min_samples_leaf=2)

    elif classifier.upper() == 'QDA':
        CLSFR = QDA(reg_param=j)

    elif classifier.lower() == 'svc':
        model = SVC(gamma='scale', random_state=5, probability=True, degree=j)

    elif clf.lower() == 'abc':
        model = ABC(base_estimator=RFC(n_estimators=int(50*j)),
                    random_state=5)

    return CLSFR


def run_classifier_model(df, cv, all_cols, clsfr, j, z):
    dt = datetime.datetime.now().date()
    CLS_df = DataFrame()

    split_dict = {0.50: [1]*cv, 0.25: [1]*cv, 0.10: [1]*cv}

    for k, v in split_dict.items():
        start_row_order = 0
        for _ in v:
            print(round(k, 3), start_row_order)
            df_s = partial_df(df, k, start_row_order, cv, rando='n')
            df_s, y = bin_df_get_y(df_s)
            for clm in all_cols:
                results = {}
                X_trn, X_tst, y_trn, y_tst, X = xy_custom(df_s, y, 0.85, clm)
                CLSFR = get_classifier(clsfr, df_s, j, z).fit(X_trn, y_trn)
                try:
                    y_score = CLSFR.predict_proba(X_tst)
                    # y_score = [num[::-1] for num in y_score]
                    print(y_score)
                    print(y_tst)
                    sleep(10)
                    y_pred = [num[-1] for num in y_score]
                    results['auc_sc'] = round(RAS(y_tst, y_pred,
                                                  average='weighted'), 4)
                    results['brier_lss'] = round(BSL(y_tst, y_pred), 4)
                    results['log_lss'] = round(LL(y_tst, y_score), 4)
                    print(results['log_lss'])

                    y_pred = [int(round(num)) for num in y_pred]
                except:
                    y_pred = CLSFR.predict(X_tst)
                    y_pred = [1 if num > 0.5 else 0 for num in y_pred]
                    results['brier_lss'] = round(BSL(y_tst, y_pred), 4)
                    results['log_lss'] = round(LL(y_tst, y_pred), 4)

                cm = CMx(y_tst, y_pred)
                results['cols'] = clm
                results['df_len'] = len(df_s)
                results['start_row'] = int((len(df)/cv) * start_row_order)
                results['cm'] = list(cm)
                results['model'] = clsfr
                results['accu'] = round(ASc(y_tst, y_pred), 4)
                results['bal_acc'] = round(BAs(y_tst, y_pred), 4)
                results['f_acc'] = round(F1S(y_tst, y_pred), 4)
                CLS_df = CLS_df.append([results], ignore_index=True)
                gc.collect()
            start_row_order += 1

    if z > 0:
        fn = f'../data/{clsfr}{j}{z}_{dt}_results.csv'
        CLS_df.to_csv(fn, index=False)
        gc.collect()

    else:
        fn = f'../data/{clsfr}{j}_{dt}_results.csv'
        CLS_df.to_csv(fn, index=False)
        gc.collect()

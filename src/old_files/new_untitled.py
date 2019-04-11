from dummies_bins_test_train_cv import initial_df
from dummies_bins_test_train_cv import bin_df_get_y

from pandas import DataFrame as DF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import BaggingClassifier as BGC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import IsolationForest as IFc
# from sklearn.ensemble import VotingClassifier as VtC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RidgeClassifier as RdC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNNc
import warnings
import gc
gc.enable()
warnings.filterwarnings('ignore')


def run_func(clsfr, param_dict, cv, X, y, fn, pre_dis=24, njbs=-3, rts=True):
    gc.collect()
    cvmdl = GridSearchCV(clsfr, param_dict, cv=cv, pre_dispatch=pre_dis,
                         n_jobs=njbs, return_train_score=rts)
    cvmdl.fit(X, y)

    bpr = cvmdl.best_params_
    bsr = cvmdl.best_score_
    bes = cvmdl.best_estimator_

    gc.collect()
    print(bes)
    print(f'Best Score:{round(bsr*100, 2)}%\n{bpr}')

    gc.collect()
    with open('./results.txt', 'w') as f:
        f.write(f'{cvmdl.cv_results_}\n{bsr}\n{bes}')
        f.write(f'')

    df = DF(cvmdl.cv_results_)
    df.to_csv(f'../data/{fn}_all.csv')
    gc.collect()


df = initial_df('../data/use_for_predictions.csv')
df, y = bin_df_get_y(df)
ac = ['diff', 'color', 'time', 'game_time', 'weekday', 'elo', 'opp_elo',
      'game_num']
df = df[ac].copy()
X = df.values

# Linear Discriminant Analysis
ld_cls = LDA(solver='lsqr')

# Logistic Regression
lr_cls = LR(C=0.01, max_iter=50, tol=7.5e-3, class_weight=None,
            solver='saga', random_state=5, multi_class='ovr')

# KNeighbors Classifier
kn_cls = KNNc(n_neighbors=41, weights='uniform', algorithm='brute',
              metric='chebyshev')

# Ridge Classifier
rd_cls = RdC(fit_intercept=False, class_weight=None, solver='lsqr',
             random_state=5)

# Random Forest Classifier
rf_cls = RFC(n_estimators=200, max_depth=10, min_samples_split=2,
             min_samples_leaf=3, max_features=None, class_weight=None,
             criterion='entropy', random_state=5)

# Extra Trees Classifier
et_cls = ETC(criterion='entropy', min_impurity_decrease=0.0, bootstrap=True,
             max_features=None, n_estimators=100, max_depth=None,
             min_samples_split=3, min_samples_leaf=2, max_leaf_nodes=20,
             class_weight=None, random_state=5)

# Gradient Boosting Classifier
gb_cls = GBC(loss='deviance', max_features=None, learning_rate=0.125,
             n_estimators=150, min_samples_split=2, min_samples_leaf=20,
             max_depth=5, min_impurity_decrease=0.20, max_leaf_nodes=10,
             random_state=5)

# Isolation Forest
if_cls = IFc(random_state=5)
if_param = {'n_estimators': [100, 200, 300],
            'contamination': [0.05, 0.1, 0.2],
            'max_features': [0.5, 0.75, 1.0],
            'bootstrap': [True, False],
            'behaviour': ['new']}
#run_func(if_cls, if_param, 7, X, y, 'iso_frst')

# Ada Boost Classifier
ab_cls = ABC(random_state=5, algorithm='SAMME')
ab_param = {'base_estimator': [lr_cls, rd_cls, rf_cls, gb_cls, et_cls],
            'n_estimators': [10, 50, 100],
            'learning_rate': [0.5, 1.0, 2.0]}

# Bagging Classifier
bg_cls = BGC(random_state=5)
bg_param = {'base_estimator': [lr_cls, kn_cls, ld_cls, rd_cls, rf_cls, gb_cls,
                               et_cls],
            'n_estimators': [10, 50, 100],
            'max_features': [0.25, 0.5, 0.75, 1.0],
            'bootstrap': [True, False],
            'bootstrap_features': [True, False]}


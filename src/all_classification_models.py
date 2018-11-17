import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import *

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from dummies_bins_test_train_cv import get_Xy_train_test
from dummies_bins_test_train_cv import cross_validation_process

from warnings import filterwarnings
filterwarnings('ignore')

df = pd.read_csv('../data/use_for_predictions.csv')


X_train, X_test, y_train, y_test, X, y, df_clean = get_Xy_train_test(
        df, .98, .99)


# ### Linear Discriminant Analysis

LDA_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'
                                     ).fit(X_train, y_train)
LDA_scores = cross_validation_process(LDA_clf, X_test, y_test, cv=11)


# ### Quadratic Discriminant Analysis

QDA_clf = QuadraticDiscriminantAnalysis(reg_param=0.26055
                                        ).fit(X_train, y_train)
QDA_scores = cross_validation_process(QDA_clf, X_test, y_test, cv=11)


# ### Gaussian Process Classifier

gpc_rbf_clf = GaussianProcessClassifier(n_jobs=-2, n_restarts_optimizer=10,
                                        random_state=9).fit(X_train, y_train)
gpc_rbf_score = cross_validation_process(gpc_rbf_clf, X_test, y_test, cv=11)


# ### Logistic Regression


lgst_reg_clf = LogisticRegression(penalty='l2', class_weight='balanced',
                                  random_state=9, max_iter=5000, C=1e-3,
                                  solver='lbfgs', n_jobs=8, multi_class='auto'
                                  ).fit(X_train, y_train)

lgst_reg_score = cross_validation_process(lgst_reg_clf, X_test, y_test, cv=11)


# ### Logistic Regression CV

lgst_reg_cv_clf = LogisticRegressionCV(Cs=10, penalty='l2', cv=6,
                                       class_weight='balanced', random_state=9,
                                       solver='newton-cg', n_jobs=-2
                                       ).fit(X_train, y_train)
lgst_reg_cv_score = cross_validation_process(lgst_reg_cv_clf, X_test, y_test,
                                             cv=11)


# ### Ada Boost Classifier

ada_clf = AdaBoostClassifier(n_estimators=274, learning_rate=0.013,
                             random_state=9).fit(X_train, y_train)
ada_scores = cross_validation_process(ada_clf, X_test, y_test, cv=11)


# ### SGD Classifier

SGD_clf = SGDClassifier(loss='hinge', penalty='l2', max_iter=1e3,
                        shuffle=False, n_jobs=8, random_state=9,
                        class_weight='balanced').fit(X_train, y_train)
SGD_score = cross_validation_process(SGD_clf, X_test, y_test, cv=11)


# ### Random Forest Classifier

rand_frst_clf = RandomForestClassifier(n_estimators=1000, criterion='entropy',
                                       n_jobs=8, min_samples_leaf=2,
                                       random_state=9, class_weight='balanced'
                                       ).fit(X_train, y_train)

rand_frst_score = cross_validation_process(rand_frst_clf, X_test, y_test,
                                           cv=11)


# ### Ridge Classifier

ridge_clf = RidgeClassifier(class_weight='balanced', random_state=9
                            ).fit(X_train, y_train)
ridge_score = cross_validation_process(ridge_clf, X_test, y_test, cv=11)


# ### Ridge Classifier CV

ridge_cv_clf = RidgeClassifierCV(scoring='average_precision', cv=20,
                                 class_weight='balanced').fit(X_train, y_train)
ridge_cv_score = cross_validation_process(ridge_cv_clf, X_test, y_test, cv=11)


# ### K Neighbors Classifier

KNN_clf = KNeighborsClassifier(n_neighbors=19, leaf_size=88, n_jobs=8
                               ).fit(X_train, y_train)
KNN_score = cross_validation_process(KNN_clf, X_test, y_test, cv=11)


# ### Multi-layer Perceptron classifier


MLP_clf = MLPClassifier(hidden_layer_sizes=(64,), activation='logistic',
                        solver='lbfgs', alpha=0.0001, batch_size=8,
                        max_iter=5000, random_state=9, validation_fraction=0.1,
                        verbose=True).fit(X_train, y_train)

MLP_score = cross_validation_process(MLP_clf, X_test, y_test, cv=11)

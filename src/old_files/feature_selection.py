from dummies_bins_test_train_cv import initial_df
from dummies_bins_test_train_cv import xy_tt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import hinge_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import log_loss
from sklearn.metrics import matthews_corrcoef
# from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
from sklearn.metrics import zero_one_loss
from sklearn.feature_selection import f_classif


def run_trial_model():
    df = initial_df()
    X_train, X_test, y_train, y_test = xy_tt(df, 0.1)
    model = RandomForestClassifier(n_jobs=-1, n_estimators=500, random_state=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    print(round(accuracy_score(y_test, y_pred), 5))
    print(confusion_matrix(y_test, y_pred))
    print(round(f1_score(y_test, y_pred), 5))
    print(round(hamming_loss(y_test, y_pred), 5))
    print(round(hinge_loss(y_test, y_pred), 5))
    print(round(jaccard_similarity_score(y_test, y_pred), 5))
    print(round(log_loss(y_test, y_pred_prob), 5))
    print(round(matthews_corrcoef(y_test, y_pred), 5))
#    print(round(precision_recall_curve(), 5))
    print(precision_recall_fscore_support(y_test, y_pred))
#    print(round(roc_auc_score(y_test, y_pred_prob[0]), 5))
#    print(round(roc_curve(), 5))
    print(round(zero_one_loss(y_test, y_pred), 5))
    ffs = f_classif(X_train, y_train)
    print(ffs)
    print(model.feature_importances_)
    return y_test, y_pred


y_test, y_pred = run_trial_model()

## Ran all the models. Deleting these:
### 1. __Multinomial Naive-Bayes__
```
multinomial_nb_clf = MultinomialNB(alpha=1).fit(X_train, y_train)
ml_nb_scores = cross_validation_process(multinomial_nb_clf, X_test, y_test, cv=11)
```
> Average Accuracy(average_precision) = __46.67%__<br>
> Scores(average_precision) = [0.459, 0.445, 0.535, 0.499, 0.417, 0.378, 0.641, 0.542, 0.520, 0.377, 0.322]<br>
> Confusion Matrix Accuracy = __58.33%__<br>
>__Confusion Matrix__:<br>
| 45 | 36 |<br>
| 29 | 46 |


### 2. __Gaussian Naive-Bayes__
```
gaussian_nb_clf = GaussianNB().fit(X_train, y_train)
gaussian_nb_score = cross_validation_process(gaussian_nb_clf, X_test, y_test)
```
> Average Accuracy(average_precision) = __46.16%__<br>
> Scores(average_precision) = 0.460, 0.426, 0.562, 0.459, 0.442, 0.408, 0.617, 0.513, 0.508, 0.365, 0.319<br>
> Confusion Matrix Accuracy = __50.00%__<br>
>__Confusion Matrix__:<br>
| 3 | 78 |<br>
| 0 | 75 |

### 3. __One Vs Rest Classifier__
```
ovr_clf = OneVsRestClassifier(LinearSVC(penalty='l2', loss='hinge', dual=True,
                                        C=32.0, random_state=9)).fit(X_train, y_train)<br>
ovr_score = cross_validation_process(ovr_clf, X_test, y_test, cv=11)
```
> Average Accuracy(average_precision) = __53.73%__<br>
Scores(average_precision) = 0.468, 0.488, 0.570, 0.517, 0.554, 0.453, 0.876, 0.613, 0.699, 0.356, 0.315<br>
Confusion Matrix Accuracy = __51.92%__<br>
>__Confusion Matrix__ :<br>
| 81 | 0 |<br>
| 75 | 0 |


### 4. __One Vs One Classifier__
```
ovo_clf = OneVsOneClassifier(LinearSVC(penalty='l2', loss='hinge', dual=True, 
                             C=32.0, class_weight='balanced', random_state=9)
                             ).fit(X_train, y_train)
ovo_score = cross_validation_process(ovo_clf, X_test, y_test, cv=11)
```
> Average Accuracy(average_precision) = __53.73%__<br>
> Scores (average_precision) = 0.468, 0.488, 0.570, 0.517, 0.554, 0.453, 0.876, 0.613, 0.699, 0.356, 0.315<br>
> Confusion Matrix Accuracy = __51.92%__<br>
>__Confusion Matrix__:<br>
| 81 | 0 |<br>
| 75 | 0 |

### 5. __Bernoulli Naive-Bayes__
```
brnl_clf = BernoulliNB().fit(X_train, y_train)
brnl_score = cross_validation_process(brnl_clf, X_test, y_test)
```
> Average Accuracy(average_precision) = __48.87%__<br>
Scores(average_precision) = 0.465, 0.465, 0.576, 0.502, 0.464, 0.399, 0.695, 0.579, 0.517, 0.362, 0.350<br>
> Confusion Matrix Accuracy = __55.13%__<br>
>__Confusion Matrix__:<br>
| 43 | 38 |<br>
| 32 | 43 |

### 6. __Decision Tree Classifier__
```
dec_tree_clf = DecisionTreeClassifier().fit(X_train, y_train)<br>
dec_tree_score = cross_validation_process(dec_tree_clf, X_test, y_test)
```
> Average Accuracy(average_precision) = __56.05%__<br>
Scores(average_precision) = 0.580, 0.427, 0.581, 0.429, 0.540, 0.700, 0.738, 0.468, 0.500, 0.531, 0.672<br>
> Confusion Matrix Accuracy = __58.97%__<br>
>__Confusion Matrix__:<br>
| 44 | 37 |<br>
| 27 | 48 |


### 7. __Extra Tree Classifier__
```
ET_clf = ExtraTreeClassifier(criterion='entropy', random_state=9,
                             class_weight='balanced').fit(X_train, y_train)
ET_score = cross_validation_process(ET_clf, X_test, y_test)
```
> Average Accuracy(average_precision) = __51.59%__<br>
Scores(average_precision) = 0.644, 0.415, 0.454, 0.481, 0.500, 0.586, 0.653, 0.539, 0.540, 0.419, 0.445<br>
> Confusion Matrix Accuracy = __60.26%__<br>
>__Confusion Matrix__:<br>
| 47 | 34 |<br>
| 28 | 47 |


### 8. __Gradient Boosting Classifier__
```
grad_boost_clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=.1,
                                            max_depth=500000, random_state=9,
                                            n_iter_no_change=100).fit(X_train, y_train)
grad_boost_score = cross_validation_process(grad_boost_clf, X_test, y_test, cv=11)
```
> Average Accuracy(average_precision) = __73.26%__<br>
Scores(average_precision) = 0.666, 0.395, 0.818, 0.759, 0.724, 0.757, 0.877, 0.733, 0.704, 0.695, 0.931<br>
> Confusion Matrix Accuracy = __61.54%__<br>
>__Confusion Matrix__ :<br>
| 48 | 33 |<br>
| 27 | 48 |

### 9. Random Forest Classifier
```
rand_frst_clf = RandomForestClassifier(n_estimators=100, criterion='entropy', n_jobs=-2,
                                       random_state=9).fit(X_train, y_train)
rand_frst_score = cross_validation_process(rand_frst_clf, X_test, y_test, cv=11)
```
> Average Accuracy(average_precision) = __73.66%__<br>
Scores(average_precision) = 0.761, 0.638, 0.818, 0.678, 0.557, 0.811, 0.927, 0.775, 0.773, 0.623, 0.739<br>
Confusion Matrix Accuracy = __67.31%__<br>
>__Confusion Matrix__ :<br>
| 53 | 28 |<br>
| 23 | 52 |

### 10. __Extra Trees Classifier__
```
XTsC_clf = ExtraTreesClassifier(n_estimators=1000, criterion='entropy', n_jobs=-2,
                                random_state=9, class_weight='balanced').fit(X_train, y_train)
XTsC_score = cross_validation_process(XTsC_clf, X_test, y_test, cv=11)
```
> Average Accuracy(average_precision) = __67.23%__<br>
Scores(average_precision) = 0.758, 0.536, 0.779, 0.641, 0.593, 0.786, 0.792, 0.556, 0.596, 0.505, 0.855<br>
Confusion Matrix Accuracy = __67.95%__<br>
>__Confusion Matrix__ :<br>
| 53 | 28 |<br>
| 22 | 53 |
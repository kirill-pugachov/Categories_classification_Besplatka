# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:03:29 2018

@author: Kirill
"""


import shelve
import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn.ensemble import VotingClassifier



#Data
start = 1
end = 2

#кол-во заголовков объявлений в списке по ключу url
members_number = 500


def get_data_from_disk():
    db = shelve.open('bags_list')
#    a = db['marked']
    a = list(db.keys())
    db.close()
    return a


def vectorizer(start, end):
    stop_w = stopwords.words('russian')
    mass_vectorizer = TfidfVectorizer(
            ngram_range=(start, end),
            token_pattern=r'\b\w\w\w+\b',
            stop_words=stop_w,
            analyzer='word',
            use_idf=True)
    return mass_vectorizer


def vectorizer_single(bags):
    vect = vectorizer(start, end)
    vect.fit(bags)
    return vect


def full_data_by_keys(key):
    db = shelve.open('bags_list')
    temp = db[key]
    db.close()
    return temp


def get_final_list(members_number):
    result = list()
    keys_list = get_data_from_disk()
    for key in keys_list:
        full_list = full_data_by_keys(key)
        if len(full_list) >= members_number:
    #        print(I)
            for raw in full_list:
    #            print(K, len(marked[I]))
                result.append([raw[0], raw[1][0]])
    return result


def list_to_frame(final_list):
    df = pd.DataFrame(final_list)
    return df


def data_ready(final_list):
    total_dataframe = list_to_frame(final_list)
    y = total_dataframe[0]
    X = total_dataframe.drop(0, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=42,
                                                        shuffle=True)
    return X_train, X_test, y_train, y_test


def full_data_ready(final_list):
    total_dataframe = list_to_frame(final_list)
    y = total_dataframe[0]
    X = total_dataframe.drop(0, 1)
    return X, y


def X_y_split(final_list):
    total_dataframe = list_to_frame(final_list)
    y = total_dataframe[0]
    X = total_dataframe.drop(0, 1)
    return X, y


def soft_voting(df_res, y):
    print('\n')
    print('SOFT VOTING START')

#    min_max_scaler = preprocessing.MinMaxScaler()
#    df_res = min_max_scaler.fit_transform(df_res)

#    robust_scaler = preprocessing.RobustScaler()
#    df_res = robust_scaler.fit_transform(df_res)

    quantile = preprocessing.QuantileTransformer()
    df_res = quantile.fit_transform(df_res)

    clf1 = ensemble.AdaBoostClassifier()
    clf2 = MLPClassifier()  #AdaBoostClassifier()#ensemble.RandomForestClassifier(n_estimators=200, random_state=11,n_jobs=-1)
    clf3 = ensemble.GradientBoostingClassifier()  #ensemble.GradientBoostingClassifier(n_estimators=3000, learning_rate=1.1, max_depth=5, random_state=11)
    clf4 = SGDClassifier(loss='log', max_iter=100)  #SGDClassifier(max_iter=35000, tol=1e-4, shuffle=True, penalty='l2', loss='log')
    clf5 = LogisticRegression()
    clf6 = LogisticRegressionCV()
    clf7 = QuadraticDiscriminantAnalysis()
    clf8 = GaussianNB()
    clf9 = LinearDiscriminantAnalysis()
    clf10 = SVC(probability=True)
    eclf = VotingClassifier(estimators=[('ada', clf1), ('mlpc', clf2),
                                        ('gbs', clf3), ('sgdc', clf4),
                                        ('lgr', clf5), ('lrcv', clf6),
                                        ('qda', clf7), ('gnb', clf8),
                                        ('lda', clf9), ('cvc', clf10)],
                                voting='soft', weights=[1,1,1,1,1,1,1,1,1,1])

    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10, eclf],
                          ['AdaBoostClassifier', 'MLPClassifier',
                           'GradientBoosting', 'SGDClassifier',
                           'LogisticRegression', 'LogisticRegressionCV',
                           'QuadraticDiscriminantAnalysis', 'GaussianNB',
                           'LinearDiscriminantAnalysis', 'SVC', 'Ensemble']):
        scores = cross_val_score(clf, df_res, y, cv=5, scoring='accuracy')
        print("ROC_AUC scoring: %0.5f (+/- %0.5f) [%s]" % (scores.mean(), scores.std(), label))
        print('SOFT VOTING END')
    return eclf


def short_soft_voting(df_res, y):
    print('\n')
    print('SOFT VOTING')

    clf1 = ensemble.AdaBoostClassifier()
    clf2 = ensemble.RandomForestClassifier(n_estimators=200, random_state=42,n_jobs=-1)
    clf3 = ensemble.GradientBoostingClassifier(n_estimators=200, learning_rate=1, max_depth=5, random_state=42)
    clf4 = SGDClassifier(loss='log', max_iter=100)  #SGDClassifier(max_iter=35000, tol=1e-4, shuffle=True, penalty='l2', loss='log')
    clf5 = LogisticRegression()
    clf6 = GaussianNB()
    clf7 = SVC(probability=True)
    eclf = VotingClassifier(estimators=[('ada', clf1), ('rndfor', clf2),
                                        ('gbs', clf3), ('sgdc', clf4),
                                        ('lgr', clf5), ('gnb', clf6),
                                        ('cvc', clf7)],
                                voting='soft', weights=[1,1,1,1,1,1,1])

    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, eclf],
                          ['AdaBoostClassifier', 'RandomForestClassifier',
                           'GradientBoostingClassifier', 'SGDClassifier',
                           'LogisticRegression', 'GaussianNB',
                           'SVC', 'Ensemble']):
        scores = cross_val_score(clf, df_res, y, cv=5, scoring='accuracy')
        print("ROC_AUC scoring: %0.5f (+/- %0.5f) [%s]" % (scores.mean(), scores.std(), label))
        print('SOFT VOTING END')
    return eclf


def classifiers_evaluation(df_res, y):
    vect = vectorizer(start, end)
    vect.fit(df_res[1])

    classifiers = [
        KNeighborsClassifier(3),
        SVC(probability=True),
        DecisionTreeClassifier(),
        ensemble.RandomForestClassifier(),
        ensemble.AdaBoostClassifier(),
        ensemble.GradientBoostingClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        LogisticRegression(),
        MLPClassifier(),
        SGDClassifier(loss='log', max_iter=100),
        LogisticRegressionCV()
        ]

    log_cols = ["Classifier", "ROC_AUC score"]
    log = pd.DataFrame(columns=log_cols)

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

    acc_dict = {}

    for train_index, test_index in sss.split(df_res, y):
        X_train, X_test = df_res.iloc[train_index], df_res.iloc[test_index]

        y_train, y_test = y[train_index], y[test_index]

        X_train_ready = vect.transform(X_train[1])

        X_test_ready = vect.transform(X_test[1])

        del X_train
        del X_test

        for clf in classifiers:
            name = clf.__class__.__name__

            clf.fit(X_train_ready, y_train)
            train_predictions = clf.predict(X_test_ready)
            acc = accuracy_score(y_test, train_predictions)
#            acc = roc_auc_score(y_test, train_predictions)

            if name in acc_dict:
                acc_dict[name] += acc
            else:
                acc_dict[name] = acc

        del X_train_ready
        del X_test_ready
        del y_train
        del y_test

    for clf in acc_dict:
        acc_dict[clf] = acc_dict[clf] / 10.0
        log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
        log = log.append(log_entry)

    print(acc_dict)
    print(log)


def short_classifiers_evaluation(df_res, y):
    vect = vectorizer(start, end)
    vect.fit(df_res[1])

    classifiers = [
        ensemble.GradientBoostingClassifier(n_estimators=150),
        ensemble.RandomForestClassifier(n_estimators=150, n_jobs=-1),
        SVC(probability=True),
        LogisticRegression(),
        SGDClassifier(loss='log', max_iter=150, n_jobs=-1)
        ]

    log_cols = ["Classifier", "Accuracy score"]
    log = pd.DataFrame(columns=log_cols)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    acc_dict = {}

    for train_index, test_index in sss.split(df_res, y):
        X_train, X_test = df_res.iloc[train_index], df_res.iloc[test_index]

        y_train, y_test = y[train_index], y[test_index]

        X_train_ready = vect.transform(X_train[1])

        X_test_ready = vect.transform(X_test[1])

        del X_train
        del X_test

        for clf in classifiers:
            name = clf.__class__.__name__

            clf.fit(X_train_ready, y_train)
            train_predictions = clf.predict(X_test_ready)
            acc = accuracy_score(y_test, train_predictions)
#            acc = roc_auc_score(y_test, train_predictions)

            if name in acc_dict:
                acc_dict[name] += acc
            else:
                acc_dict[name] = acc

        del X_train_ready
        del X_test_ready
        del y_train
        del y_test

    for clf in acc_dict:
        acc_dict[clf] = acc_dict[clf] / 5.0
        log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
        log = log.append(log_entry)

    print(acc_dict)
    print(log)


if __name__ == '__main__':
    final_list = get_final_list(members_number)[0:20000]

    X, y = full_data_ready(final_list)
    short_classifiers_evaluation(X, y)

#    X_train, X_test, y_train, y_test = data_ready(final_list)
#    vect_1 = vectorizer(start, end)
#    vect_1.fit(X_test[1])
#    X_train_ready = vect_1.transform(X_train[1])
#    cls_ensemble = short_soft_voting(X_train_ready, y_train)




#    clf_list = [
#            LinearSVC(random_state=0, multi_class='ovr'),
#            LogisticRegression(solver='newton-cg', multi_class='ovr'),
#            SGDClassifier(max_iter=1000)
#            ]
#
#    for model in clf_list:
#        clf = OneVsRestClassifier(model)
#        temp = clf.fit(X_ready, y_train).predict(X_ready)
#        scores = cross_val_score(clf, X_ready, y_train, cv=5, n_jobs=-1, scoring='accuracy')
#        print("Cross_val_Score accuracy scoring: %0.5f (+/- %0.5f) [%s]" % (scores.mean(), scores.std(), model.__class__))
#        temp_0 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_ready, y_train).predict(vect.transform(X_test[1]))
#        scores1 = cross_val_score(clf, vect.transform(X_test[1]), y_test, cv=5, n_jobs=-1, scoring='accuracy')
#        print("Cross_val_Score accuracy scoring: %0.5f (+/- %0.5f) [%s]" % (scores1.mean(), scores1.std(), model.__class__))

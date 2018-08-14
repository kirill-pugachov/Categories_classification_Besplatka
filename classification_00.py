# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:03:29 2018

@author: Kirill
"""


import shelve
import pandas as pd
#import numpy as np
#from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multiclass import OneVsRestClassifier
#from sklearn.svm import LinearSVC
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

#from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
#from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib



#Data
start = 1
end = 1
shelve_name = 'bags_list'
#кол-во заголовков объявлений в списке по ключу url
members_number = 1500
#кол-во категорий в одной модели
size = 50
#хранилище списка имен обученных моделей
models_shelve_name = 'models_list'
key_for_models = 'models'
url_models_file_name = 'url_in_models_used'

def write_to_disk_file(order, url_models_file_name):
    with open(url_models_file_name + '.txt', 'w', newline='') as log_file:
        for orde in order:
            log_file.write(str(orde)+'\n') 
            

def creat_on_disk(data, key, shelve_name):
    db = shelve.open(shelve_name)
    db[key] = data
    db.close()
    pass


def write_to_disk(data, key, shelve_name):
    db = shelve.open(shelve_name)
    db[key].append(data)
    db.close()
    pass


def get_key_dfd(shelve_name):
    '''
    Получает ключи из внешнего хранилища
    по адресу shelve_name
    '''
    db = shelve.open(shelve_name)
    a = list(db.keys())
    db.close()
    return a
    
    
def get_data_from_disk(shelve_name, key):
    '''
    По имени хранилища shelve_name и ключу key
    получает списки заголовков запросов и меток
    из внешнего хранилища
    '''
    db = shelve.open(shelve_name)
    a = db[key]
    db.close()
    return a
    

def get_keys_list(shelve_name, members_number):
    '''
    Получаем список ключей в shelve
    отбираем с заданным размером значений
    '''
    res = list()
    db = shelve.open(shelve_name)
    for key in list(db.keys()):
        if len(get_data_from_disk(shelve_name, key)) >= members_number:
            res.append(key)
    write_to_disk_file(res, url_models_file_name)
    return res


def build_keys_list_generator(shelve_name):
    '''
    Заготовка генератора списка ключей
    '''
    for key in get_keys_list(shelve_name, members_number):
        yield key
    
    
def build_final(shelve_name, size):
    '''
    Делим список ключей на части по заданному размеру
    '''
    res = list()
    generator = build_keys_list_generator(shelve_name)
    a = len(get_keys_list(shelve_name, members_number))
    for _ in range(round(a/size)):
        temp = list()
        for __ in range(size):
            try:
                temp.append(next(generator))
            except StopIteration:
                res.append(temp)
                return res                 
        res.append(temp)
    return res


def get_final_list(keys_list):
    '''
    Формирует финальный список списков
    [..
    [метка (url категории), заголовок],
    [метка (url категории), заголовок],
    ...]
    '''
    result = list()
    for key in keys_list:
        for raw in get_data_from_disk(shelve_name, key):
            result.append([raw[0], raw[1][0]])
    return result


def vectorizer(start, end):
    '''
    Делает векторизатор, где start, end определяет
    кол-во слов в ngram_range
    '''
    stop_w = stopwords.words('russian')
    mass_vectorizer = TfidfVectorizer(
            ngram_range=(start, end),
            token_pattern=r'\b\w\w\w+\b',
            stop_words=stop_w,
            analyzer='word') #,
#            use_idf=True)
    return mass_vectorizer


def list_to_frame(final_list):
    df = pd.DataFrame(final_list)
    return df


def data_ready(final_list):
    '''
    Разбивает финльный список на метки и значения
    '''
    total_dataframe = list_to_frame(final_list)
    y = total_dataframe[0]
    X = total_dataframe.drop(0, 1)
    return X, y


def build_models(shelve_name, size):
    res = list()
    keys_lists = build_final(shelve_name, size)
    for keys_list in keys_lists:
        final_list = get_final_list(keys_list)
        X, y = data_ready(final_list)
        vect = vectorizer(start, end)
        vect.fit(X[1])
        X_ready = vect.transform(X[1])
#        cls = OneVsRestClassifier(
#                LinearSVC(random_state=0, multi_class='ovr')
#                ).fit(X_ready, y)
        cls = OneVsRestClassifier(SGDClassifier(max_iter=100, loss='log')).fit(X_ready, y)
        name = str(keys_lists.index(keys_list)) + '_model.pkl'
        joblib.dump((cls, vect), name)
        res.append(name)
    creat_on_disk(res, key_for_models, models_shelve_name)
    return res


if __name__ == '__main__':
    model_list = build_models(shelve_name, size)
    
    
    
    
    

##from sklearn.externals import joblib
## now you can save it to a file
#joblib.dump(clf, 'filename.pkl') 
## and later you can load it
#clf = joblib.load('filename.pkl')        
#    
#    final_list = get_final_list(shelve_name)
#    X_train, X_test, y_train, y_test = data_ready(final_list)
#    vect = vectorizer(start, end)
#    vect.fit(X_test[1])
#    X_ready = vect.transform(X_train[1])
#    clf_list = [
#            LinearSVC(random_state=0, multi_class='ovr'),
#            LogisticRegression(solver='newton-cg', multi_class='ovr'),
#            SGDClassifier(max_iter=1000)
#            ]
#    
#    for model in clf_list:
#        clf = OneVsRestClassifier(model)
##        temp = clf.fit(X_ready, y_train).predict(X_ready)
#        scores = cross_val_score(clf, X_ready, y_train, cv=5, n_jobs=-1, scoring='accuracy')
#        print("Cross_val_Score accuracy scoring: %0.5f (+/- %0.5f) [%s]" % (scores.mean(), scores.std(), model.__class__))
##        temp_0 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_ready, y_train).predict(vect.transform(X_test[1]))
#        scores1 = cross_val_score(clf, vect.transform(X_test[1]), y_test, cv=5, n_jobs=-1, scoring='accuracy')
#        print("Cross_val_Score accuracy scoring: %0.5f (+/- %0.5f) [%s]" % (scores1.mean(), scores1.std(), model.__class__))
#
#def split_keys_list(shelve_name, size):
#    result = list()
#    generator = build_keys_list_generator(shelve_name)
#    try:
#        for _ in range(size):
#            result.append(next(generator))
#    except StopIteration:
#        return result
#    return result
#    
#def vectorizer_single(bags):
#    vect = vectorizer(start, end)
#    vect.fit(bags)
#    return vect
#
#
#def list_to_frame(final_list):
#    df = pd.DataFrame(final_list)
#    return df


#def X_y_split(final_list):
#    total_dataframe = list_to_frame(final_list)
#    y = total_dataframe[0]
#    X = total_dataframe.drop(0, 1)
#    return X, y


#def get_final_list():
#    result = list()
#    marked = get_data_from_disk()
#    for I in range(len(marked)):
##        print(I)
#        for K in range(len(marked[I])):
##            print(K, len(marked[I]))
#            result.append([marked[I][K][0], marked[I][K][1][0]])
#    return result
#def build_models(shelve_name, size):
#    res = list()
#    creat_on_disk(res, key, shelve_name)
#    keys_lists = build_final(shelve_name, size)
#    for keys_list in keys_lists:
#        final_list = get_final_list(keys_list)
#        X, y = data_ready(final_list)
#        vect = vectorizer(start, end)
#        vect.fit(X[1])
#        X_ready = vect.transform(X[1])
#        cls = OneVsRestClassifier(LinearSVC(random_state=0, multi_class='ovr')).fit(X_ready, y)
#        res.append((vect, cls))
#        write_to_disk(res, key, models_shelve_name)
#    return res
    
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 09:50:15 2018

@author: User
"""

import shelve
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib


#хранилище списка имен обученных моделей
models_shelve_name = 'models_list'
key_for_models = 'models'
##хранилище для векторизатора
#vector_shelve_name = 'vectorizer'
#key_for_vector = 'vector'


def get_models_list(models_shelve_name, key_for_models):
    db = shelve.open(models_shelve_name)
    temp = db[key_for_models]
    db.close()
    return temp


def get_model(model_key):
    cls, vect = joblib.load(model_key)
    return cls, vect


if __name__ == '__main__':
    models_list = get_models_list(models_shelve_name, key_for_models)
    for model in models_list:
        _, vect = get_model(model)
        print(model, '\n', len(vect.get_feature_names()), '\n', vect.get_feature_names())

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:56:12 2018

@author: Kirill
"""

import shelve
#import pandas as pd
#import numpy as np
#from sklearn.model_selection import train_test_split
#from nltk.corpus import stopwords
#from sklearn.feature_extraction.text import TfidfVectorizer

#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.svm import LinearSVC
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import SGDClassifier

#from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
#from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib



#Data
start = 1
end = 1
shelve_name = 'bags_list'
#хранилище списка имен обученных моделей
models_shelve_name = 'models_list'
key_for_models = 'models'
test_words = [
        'Volkswagen Golf IV 1.4i 2003',
        'Балка подвески передняя задняя Кадди Кадді 04-09 разборка',
        'Стекло скло окно Партнер Берлінго Берлинго 03-08 09-12 разборка М59 В9',
        'Дзеркало зеркало Партнер Берлинго Берлінго -08 -12 М59 В9 разборка',
        'Штани для хлопчика мужские РОЗПРОДАЖ штаны мальчика джинси джинсові',
        'Нова чорна шкіряна спідничка, спідниця, юбка, міні, кожаная юбка, кожа',
        'Флісовий демісезонний комбінезон 68 / флисовый деми комбинезон',
        'Leagoo M9 IPS:5,5-2Gb/16Gb-8Мрх -безрамочный 18:9 2850 mAh НОВИНКА',
        'Apple iPhone 6 16 GB Space Gray/Silver Neverlock',
        'Xiaomi Redmi Note 5a Prime Grey,Gold,Pink 3/32 и 4/64 +Бампер и Стекло',
        'Задняя панель корпуса (крышка) LG G4 H818 (серая)',
        'Iphone/Айфон 5s 16 GB Spase Gray/Gold/Silver •Оригінальний комплект!',
        'Leagoo M9 IPS:5,5-2Gb/16Gb-8Мрх -безрамочный 18:9 2850 mAh НОВИНКА',
        'Xiaomi Redmi 5plus 3/32 Global Version Золотий, Чорний',
        'ZTE Nubia M2 4/64 GB AMOLED Чёрный без предоплаты ГЛОБАЛ версия',
        'Мощный ИГРОВОЙ ПК / 3.8-4.2Ghz / GeForce 2GB GDDR5 ОС / РАССРОЧКА!',
        'Райзер карта 1x в 16x с USB 3.0, 6-pin, VER 007, Есть в наличии',
        'Тихий компьютерный корпус Zalman z5 plus ATX с окном СОСТОЯНИЕ НА 5',
        'Intel Pentium G630 G645 G840 G2020 Xeon E3-1220 X3430 s1155/s1156',
        'Оперативна пам\'ять Kingston 2GB DDR2-667 PC2-5300U для Intel/AMD',
        'Кофеварка Saeco Magik(Viena) Б/У из Германии',
        'Кофемашина оптом Кавовий апарат гуртом',
        'Посудомойка полная встройка 60см. Посудомоечная машина IKEA Electrolux',
        'Квартира з дизайнерським ремонтом 1 км Білогірська',
        '2х кімнатна квартира вул. Старий ринок 3(власник)',
        'Квартира на якій Ви реально економите Власник, 1кім. Київська'
        ]

def get_models(models_shelve_name, key_for_models):
    db = shelve.open(models_shelve_name)
    temp = db[key_for_models]
    db.close()
    return temp


def get_result(models_shelve_name, key_for_models, test_words):
    model_list = get_models(models_shelve_name, key_for_models)
    result = list()
    for word in test_words:
        res = list()
        for model in model_list:
            cls, vect = joblib.load(model)
            res.append(
                    (word,
                     max(cls.predict_proba(vect.transform([word])).tolist()[0]),
                     cls.predict(vect.transform([word]))[0])
                    )
        result.append(res)
    return result


def fast_get_result(models_shelve_name, key_for_models, test_words):
    model_list = get_models(models_shelve_name, key_for_models)
    vect = get_models(vector_shelve_name, key_for_vector)
    result = list()
    for word in test_words:
        res = list()
        for model in model_list:
            cls = joblib.load(model)
            res.append(
                    (word,
                     max(cls.predict_proba(vect.transform([word])).tolist()[0]),
                     cls.predict(vect.transform([word]))[0])
                    )
        result.append(res)
    return result


def res_print(result):
    for item in result:
        print('\n')
        for it in item:
            print(it[0], it[1], it[2])


if __name__ == '__main__':
#    result = get_result(models_shelve_name, key_for_models, test_words)
#    res_print(result)
    while True:
        word = [input("Введите строку для определения категории: ")]
        if word != ['stop']:
            result = get_result(models_shelve_name, key_for_models, word)
            temp = sorted(result[0], key=lambda x: x[1], reverse=True)
            res_print([temp[0:6]])
        elif word == ['stop']:
            break




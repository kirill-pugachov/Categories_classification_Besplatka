# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:04:17 2018

@author: Kirill
"""

#
#Список категорий на входе
#По одной урл из списка, собираем заголовки объявлений
#Собранное сохраняем в пикл, чтобы зря не скрапить сайт
#Тут же определяем кол-во страниц в категории
#Все одной категории заголовки делим на тестовую и обучающую выборки
#Подбираем лучший классификатор из scikit-learn
#Дообучаем модель тестовой выборкой
#Сохраняем веторизатор и модель категории в пикл
#
#Берем заголовки внешнего источника и проверяем по имеющимся связкам
#векторизатор-классификатор; проверяем полученную точность
#делаем выводы
#

import shelve
import requests
from lxml import html
from sitemap_parser import result_data

#Data
shelve_name = 'bags_list'
url_list = result_data()[715:875]
print(len(url_list))

#url_list = [
#        'https://besplatka.ua/transport',
#        'https://besplatka.ua/electronika-i-bitovaya-tehnika/smartfone-telefone',
#        'https://besplatka.ua/electronika-i-bitovaya-tehnika/smartfone-telefone/acesuari-dlya-telephonov',
#        'https://besplatka.ua/electronika-i-bitovaya-tehnika/kompyutery-i-komplektuyushie',
#        'https://besplatka.ua/electronika-i-bitovaya-tehnika/kompyutery-i-komplektuyushie/nastolnye-kompyuteri',
#        'https://besplatka.ua/electronika-i-bitovaya-tehnika/kompyutery-i-komplektuyushie/noutbuki',
#        'https://besplatka.ua/odezhda-obuv-aksessuary/zhenskaya-odezhda',
#        'https://besplatka.ua/detskiy-mir/detskaya-obuv']


#url_list = [
#        'https://besplatka.ua/electronika-i-bitovaya-tehnika/kompyutery-i-komplektuyushie/nastolnye-kompyuteri',
#        'https://besplatka.ua/electronika-i-bitovaya-tehnika/kompyutery-i-komplektuyushie/noutbuki',
#        'https://besplatka.ua/odezhda-obuv-aksessuary/zhenskaya-odezhda',
#        'https://besplatka.ua/detskiy-mir/detskaya-obuv']


def tags_to_list(header_tag):
    '''Возвращаем тайтлы списком списков'''
    result = list()
    if header_tag:
        for tag in header_tag:
            result.append([str(tag).strip()])
        return result
    else:
        return result.append(['no_tag'])


def get_ads_title_list(parsed_body):
    '''
    Получаем заголовки объявлений
    из категории
    '''
    try:
        ads_title = parsed_body.xpath('//div[@class="title"]/a/text()')
#        print(tags_to_string(ads_title))
    except:
        ads_title = ['no ads at all']
#        print(tags_to_string(ads_title))
    return tags_to_list(ads_title)


def add_mark_to_list(mark, tags_list):
    result = list()
    if tags_list:
        for tag in tags_list:
            result.append([mark, tag])
    return result


def check_page_numbers(parsed_body):
    '''
    Определяем сколько всего страниц в категории
    '''
    page_nambers = parsed_body.xpath('//li[@class="last"]/a/text()')
    if len(page_nambers):
        if int(page_nambers[0]) >= 0:
            return int(page_nambers[0])
        else:
            return 0
    else:
        return 0


def gen_new_url(url, number):
    '''
    Генерим все урл страниц в категории
    по данным числа пагинации взятого с
    первой страницы категории
    '''
    result = list()
#    result.append(url)
    if number:
        for item in range(2, number + 1):
            result.append(url + '/page/' + str(item))
        if len(result):
            return result
        else:
            return []
    else:
        return []


def get_data(url):
    response = requests.get(url)
    parsed_body = html.fromstring(response.text)
#    ads_title_bag = get_ads_title(parsed_body)
    ads_title_list = add_mark_to_list(url_list.index(url), get_ads_title_list(parsed_body))
    page_numbers = check_page_numbers(parsed_body)
    if page_numbers:
        pagination_list = gen_new_url(url, page_numbers)
        for page in pagination_list:
            response = requests.get(page)
            parsed_body = html.fromstring(response.text)
#            ads_title_bag += ' '
#            ads_title_bag += get_ads_title(parsed_body)
            ads_title_list += add_mark_to_list(url_list.index(url), get_ads_title_list(parsed_body))
    return ads_title_list #ads_title_bag,


def read_data(url_list):
#    result_bags = list()
    result_list = list()
    for url in url_list:
        ads_title_list = get_data(url) #bag_of_wards,
#        result_bags.append(bag_of_wards)
        result_list.append(ads_title_list)
    return result_list #result_bags,


def get_data_full(parsed_body, url):
    ads_title_list = add_mark_to_list(url, get_ads_title_list(parsed_body))
    return ads_title_list


def read_data_full(url_list):
    while True:
        try:
            url = url_list.pop(0)
        except:
            print('Sitemap закончился - категорий больше нет')
            break
        else:
            result_list = list()
            if check_url_on_disk(shelve_name, url):
                response = requests.get(url)
                if response.status_code == 200:
                    parsed_body = html.fromstring(response.text)
                    page_numbers = check_page_numbers(parsed_body)
                    if page_numbers:
                        result_list += get_data_full(parsed_body, url)
                        add_urls = gen_new_url(url, page_numbers)
                        while True:
                            try:
                                add_url = add_urls.pop(0)
                            except:
                                print('В данной категории закончились страницы пагинации')
                                break
                            else:
                                add_response = requests.get(add_url)
                                if add_response.status_code == 200:
                                    add_parsed_body = html.fromstring(add_response.text)
                                    result_list += get_data_full(add_parsed_body, url)
                                else:
                                    add_urls.append(add_url)
                                    continue
                    else:
                        result_list += get_data_full(parsed_body, url)
                else:
                    url_list.append(url)
            else:
                continue
        write_to_disk(result_list, url, shelve_name)
        del result_list
        print('Осталось категорий для скраппинга: ', len(url_list))
    pass


def write_to_disk(marked_data_list, url, shelve_name):
    db = shelve.open(shelve_name)
    db[url] = marked_data_list
    db.close()
    pass


def read_data_full_new(url_list):
    while True:
        try:
            url = url_list.pop(0)
        except:
            print('Sitemap закончился - категорий больше нет')
            break
        else:
            result_list = list()
            response = requests.get(url)
            if response.status_code == 200:
                parsed_body = html.fromstring(response.text)
                page_numbers = check_page_numbers(parsed_body)
                if page_numbers:
                    result_list += get_data_full(parsed_body, url)
                    add_urls = gen_new_url(url, page_numbers)
                    while True:
                        try:
                            add_url = add_urls.pop(0)
                        except:
                            print('В данной категории закончились страницы пагинации')
                            break
                        else:
                            add_response = requests.get(add_url)
                            if add_response.status_code == 200:
                                add_parsed_body = html.fromstring(add_response.text)
                                result_list += get_data_full(add_parsed_body, url)
                            else:
                                add_urls.append(add_url)
                                continue
                else:
                    result_list += get_data_full(parsed_body, url)
            else:
                url_list.append(url)
        write_to_disk_new(result_list, url, shelve_name)
        del result_list
        print('Осталось категорий для скраппинга: ', len(url_list))
    pass


def write_to_disk_new(marked_data_list, url, shelve_name):
    db = shelve.open(shelve_name)
    if url in db.keys():
        db[url] = clean_dubles(db[url] + marked_data_list, url)
    else:
        db[url] = clean_dubles(marked_data_list, url)
    db.close()
    pass


def clean_dubles(result_list, url):
    temp_set = set()
    for raw in result_list:
        temp_set.add(raw[1][0])
    result = [[url, [t]] for t in temp_set]
    return result


def check_url_on_disk(shelve_name, url):
    res = 1
    db = shelve.open(shelve_name)
    if url in list(db.keys()):
        res = 0
    db.close()
    return res


if __name__ == '__main__':
    read_data_full_new(url_list)
#    marked_data_list = read_data(url_list)
#    marked_data_list = read_data_full(url_list)
#    write_to_disk(marked_data_list)


#def tags_to_string(header_tag):
#    string = ''
##    print(len(header_tag))
#    if len(header_tag) > 1:
#        for tag in header_tag:
#            if len(tag.strip()) > 5:
#                string += str(tag).strip() + ' '
#            else:
#                string += 'no_tag_'
##        print(string)
#        return string
#    elif len(header_tag) == 1:
#        if len(header_tag[0].strip()) > 5:
#            string = str(header_tag[0].strip())
#        else:
#            string = 'no_tag'
#        return string.strip()
#    elif len(header_tag) == 0:
#        string = 'no_tag'
#        return string.strip()

#def get_ads_title(parsed_body):
#    '''
#    Получаем заголовки объявлений
#    из категории
#    '''
#    try:
#        ads_title = parsed_body.xpath('//div[@class="title"]/a/text()')
##        print(tags_to_string(ads_title))
#    except:
#        ads_title = ['no ads at all']
##        print(tags_to_string(ads_title))
#    return tags_to_string(ads_title)
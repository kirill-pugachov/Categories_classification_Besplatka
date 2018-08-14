# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:53:11 2018

@author: Kirill
"""


import requests
import copy
from random import shuffle
from lxml import html


#Data
sitemap_url = 'https://besplatka.ua/sitemaps/sitemap-besplatka.xml'
size = 8

def get_url_sitemap(sitemap_url):
    res = requests.get(sitemap_url)
    sub_res = html.fromstring(res.content).xpath('//url/loc/text()')
    return sub_res


def last_level_urls(url_list):
    result = list()
    url_list = sorted(url_list)
    ul_carbon = copy.deepcopy(url_list)
    url = url_list.pop(0)
    depends = 0
    while url_list:
        for item in ul_carbon:
            if url in item:
                depends += 1
#                print(url, item, depends)
        if depends > 1:
            url = url_list.pop(0)
            depends = 0
        elif depends == 1:
            result.append(url)
            url = url_list.pop(0)
            depends = 0
#            print(len(result))
    return shuffle_list(result)


def shuffle_list(low_level_url):
    shuffle(low_level_url)
    return low_level_url


def read_url_row(result):
    for url in result:
        yield url


def read_url_batch(result, size):
    url_blocks = dict()
    generator = read_url_row(result)
    for i in range(round(len(result)/size)):
        try:
            for _ in range(size):
                if i in url_blocks:
                    url_blocks[i].append(next(generator))
                else:
                    url_blocks[i] = [next(generator)]
        except StopIteration:
            return url_blocks
    return url_blocks


def result_data():
    url_list = get_url_sitemap(sitemap_url)
    low_level_url = last_level_urls(url_list)
#    result_dict = read_url_batch(low_level_url, size)
    return low_level_url
    

def write_to_disk(order):
    with open('url_list.txt', 'a', newline='') as log_file:
        for orde in order:
            log_file.write(str(orde)+'\n') 

                           
if __name__ == '__main__':
    url_to_work = result_data()

                
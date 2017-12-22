# -*- coding: utf-8 -*-
import os
import sys
import re
import time
start = time.time()

from gensim import corpora, matutils
import MeCab

DATA_DIR_PATH = './data/text/'
DICTIONARY_FILE_NAME = 'livedoordic.txt'
mecab = MeCab.Tagger('mecabrc')

def get_class_id(file_name):
    dir_list = get_dir_list()
    dir_name = next(filter(lambda x: x in file_name, dir_list), None)
    if dir_name:
        return dir_list.index(dir_name)
    return None

def get_dir_list():
    tmp = os.listdir(DATA_DIR_PATH)
    if tmp is None:
        return None
    return sorted([x for x in tmp if os.path.isdir(DATA_DIR_PATH + x)])

def get_file_content(file_path):
    with open(file_path, encoding='utf-8') as f:
        return ''.join(f.readlines()[2:])  # ライブドアコーパスが3行目から本文はじまってるから


def tokenize(text):
    node = mecab.parseToNode(text)
    while node:
        if node.feature.split(',')[0] == '名詞':
            yield node.surface.lower()
        node = node.next


def check_stopwords(word):
    if re.search(r'^[0-9]+$', word):  # 数字だけ
        return True
    return False


def get_words(contents):
    ret = []
    for k, content in contents.items():
        ret.append(get_words_main(content))
    return ret


def get_words_main(content):
    return [token for token in tokenize(content) if not check_stopwords(token)]


def filter_dictionary(dictionary):
    dictionary.filter_extremes(no_below=20, no_above=0.3)  # この数字はあとで変えるかも
    return dictionary


def get_contents():
    dir_list = get_dir_list()
    if dir_list is None:
        return None
    ret = {}
    for dir_name in dir_list:
        file_list = os.listdir(DATA_DIR_PATH + dir_name)

        if file_list is None:
            continue
        for file_name in file_list:
            if dir_name in file_name:  # LICENSE.txt とかを除くためです。。
                ret[file_name] = get_file_content(DATA_DIR_PATH + dir_name + '/' + file_name)
    return ret

def get_files():
    dir_list = get_dir_list()
    if dir_list is None:
        return None
    ret = {}
    for dir_name in dir_list:
        file_list = os.listdir(DATA_DIR_PATH + dir_name)
        if file_list is None:
            continue
        for file_name in file_list:
            if dir_name in file_name:  # LICENSE.txt とかを除くためです。。
                print(dir_name)
                # ret[file_name] = get_file_content(DATA_DIR_PATH + dir_name + '/' + file_name)
    # return ret

def get_vector(dictionary, content):
    tmp = dictionary.doc2bow(get_words_main(content))
    dense = list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
    return dense


def get_dictionary(create_flg=False, file_name=DICTIONARY_FILE_NAME):
    if create_flg or not os.path.exists(file_name):
        contents = get_contents()
        words = get_words(contents)
        dictionary = filter_dictionary(corpora.Dictionary(words))
        if file_name is None:
            sys.exit()
        dictionary.save_as_text(file_name)
    else:
        dictionary = corpora.Dictionary.load_from_text(file_name)
    return dictionary

# def get_lividoor_txt():
#     allLines = open("livedoor_train_c4.txt").read()


if __name__ == '__main__':
    
    dictionary = get_dictionary(create_flg=False)
    key_test_list, item_test_list = [], []
    text_test_list = []
    for line in open('livedoor_test_c4.txt', 'r'):
        r = re.split(' , ', line)
        key_test_list.append(re.search('[0-9]', r[0]).group(0))
        item_test_list.append(get_vector(dictionary, r[1]))
        text_test_list.append(r[1])

    import pickle
    from sklearn.cluster import KMeans

    print("predict !")

    print("key:", key_test_list)
    filename = 'livedoor_kmeans2_model'
    KMeans = pickle.load(open(filename, 'rb'))
    print(KMeans.predict(item_test_list))

    filename = 'livedoor_kmeans3_model'
    KMeans = pickle.load(open(filename, 'rb'))
    print(KMeans.predict(item_test_list))

    filename = 'livedoor_kmeans4_model'
    KMeans = pickle.load(open(filename, 'rb'))
    print(KMeans.predict(item_test_list))
    
    # filename = 'livedoor_kmeans5_model'
    # KMeans = pickle.load(open(filename, 'rb'))
    # print(KMeans.predict(item_test_list))
    
    print("================")
    for (i, k) in zip(range(11) ,KMeans.predict(item_test_list)):
        print(k)
        print(text_test_list[i])

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    #-----------------------------------------------
    # allLines = open("data/dokujo1.txt").read()
    # dictionary = get_dictionary(create_flg=False)
    # print(get_vector(dictionary, allLines))


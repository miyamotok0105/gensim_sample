# -*- coding: utf-8 -*-
import os
import sys
import re
from gensim import corpora, matutils
import MeCab
from sklearn.model_selection import train_test_split

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
    if re.search(r'^[0-9]+$', word):
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
    dictionary.filter_extremes(no_below=20, no_above=0.3) 
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

def fit_svm(x_train, y_train):
    from sklearn import svm
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    return accuracy_score(x_train, y_train)

# def get_lividoor_txt():
#     allLines = open("livedoor_train_c4.txt").read()


if __name__ == '__main__':

    dictionary = get_dictionary(create_flg=False)
    key_list, item_list = [], []
    for line in open('livedoor_all_c4.txt', 'r'):
        r = re.split(' , ', line)
        key_list.append(re.search('[0-9]', r[0]).group(0))
        item_list.append(get_vector(dictionary, r[1]))

    key_train_list, item_train_list = [], []
    key_test_list, item_test_list = [], []
    key_train_list, key_test_list, item_train_list, item_test_list = train_test_split(key_list, item_list, test_size=0.33, random_state=None)
    
    print("train======")
    print(len(key_train_list))
    print(len(item_train_list))
    print("test======")
    print(len(key_test_list))
    print(len(item_test_list))
    print(key_train_list[0])
    print(key_train_list[1])
    print(key_train_list[2])
    print(key_test_list[0])
    print(key_test_list[1])
    print(key_test_list[2])



    import pickle
    from sklearn import svm
    svc = svm.SVC()
    training_x = item_train_list
    training_y = key_train_list

    # # training_xは、BOWでベクトル化した各文書のリスト
    # # training_yは、文書のカテゴリのリスト（ラベル）
    svc.fit(training_x, training_y)

    filename = 'livedoor_svm_model'
    pickle.dump(svc, open(filename, 'wb'))

    print(svc.fit(training_x, training_y).score(item_test_list, key_test_list))
    # print(svc.predict_proba(item_test_list))

    from sklearn.ensemble import RandomForestClassifier
    random_forest = RandomForestClassifier()
    print(random_forest.fit(training_x, training_y).score(item_test_list, key_test_list))
    # print(random_forest.predict_proba(item_test_list))

    filename = 'livedoor_forest_model'
    pickle.dump(random_forest, open(filename, 'wb'))
    
    # from sklearn.externals.joblib import Parallel, delayed
    # results = Parallel(n_jobs=4)(delayed(fit_svm)
    # print(results)
    

    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=2, random_state=0)
    # print(kmeans.fit(training_x).score(item_test_list, key_test_list))
    # filename = 'livedoor_kmeans2_model'
    # pickle.dump(kmeans, open(filename, 'wb'))

    # kmeans = KMeans(n_clusters=3, random_state=0)
    # print(kmeans.fit(training_x).score(item_test_list, key_test_list))
    # filename = 'livedoor_kmeans3_model'
    # pickle.dump(kmeans, open(filename, 'wb'))

    # kmeans = KMeans(n_clusters=4, random_state=0)
    # print(kmeans.fit(training_x).score(item_test_list, key_test_list))
    # filename = 'livedoor_kmeans4_model'
    # pickle.dump(kmeans, open(filename, 'wb'))

    # kmeans = KMeans(n_clusters=5, random_state=0)
    # print(kmeans.fit(training_x).score(item_test_list, key_test_list))
    # filename = 'livedoor_kmeans5_model'
    # pickle.dump(kmeans, open(filename, 'wb'))

    #-----------------------------------------------
    # allLines = open("data/dokujo1.txt").read()
    # dictionary = get_dictionary(create_flg=False)
    # print(get_vector(dictionary, allLines))


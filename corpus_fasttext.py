# -*- coding: utf-8 -*-
import os
import sys
import re
from gensim import corpora, matutils
import MeCab

DATA_DIR_PATH = './text/'
DICTIONARY_FILE_NAME = 'livedoordic.txt'
mecab = MeCab.Tagger('mecabrc')
label = "__label__"

def get_dir_list():
    tmp = os.listdir(DATA_DIR_PATH)
    if tmp is None:
        return None
    #ここ　エラー出るなら　削ったりしていじってね
    tmp.remove("CHANGES.txt")
    # tmp.remove(".DS_Store")
    tmp.remove("README.txt")
    return tmp

def get_file_list(folder):
    tmp = os.listdir(os.path.join(DATA_DIR_PATH, folder))
    if tmp is None:
        return None
    tmp.remove("LICENSE.txt")
    return tmp

def write_livedoordic(strs):
    f = open('livedoordic_fasttext.txt', 'a')
    f.writelines(strs)
    f.close()

if __name__ == '__main__':
    text_dict = {}
    dir_list = get_dir_list()
    print(dir_list)

    for i in range(len(dir_list)):
        print(dir_list[i])
        print(label +str(i+1))
        file_list = get_file_list(dir_list[i])
        for f_ix in range(len(file_list)):
            allLines = open(os.path.join("text" ,dir_list[i], file_list[f_ix])).read()
            allLines = re.sub(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)", "" ,allLines)
            allLines = allLines.replace('\n','')
            write_livedoordic("%s , "%(label +str(i+1)) + allLines + "\n")


    

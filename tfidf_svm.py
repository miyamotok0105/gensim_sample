

#coding:utf-8
#!/usr/local/bin/python

import sys;
import myutil;
from gensim import corpora, models, matutils;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.naive_bayes import GaussianNB;
from sklearn.naive_bayes import MultinomialNB;
# from sklearn import cross_validation;
import numpy as np;
from optparse import OptionParser;
from sklearn.cluster import KMeans;
from scipy.cluster.hierarchy import linkage, dendrogram;
from scipy.spatial.distance import pdist;
from matplotlib.pyplot import show;

class BoWModel:
    def __init__(self, dictionary):
        self._dictionary = dictionary;

    def to_feature_vec(self, text):
        bow = self._dictionary.doc2bow(text);
        return list(matutils.corpus2dense([bow], num_terms=len(self._dictionary)).T[0]);

    def show_topics(self, num_topics=-1):
        print(self._dictionary)

class TFIDFModel:
    def __init__(self, dictionary, datafiles):
        self._dictionary = dictionary;
        self._model = self._init_model(datafiles);

    def _init_model(self, datafiles):
        corpus = [];
        for datafile in datafiles:
            corpus.extend([self._dictionary.doc2bow(line) for line in myutil.tokenize_file(datafile)]);
        return models.TfidfModel(corpus);

    def to_feature_vec(self, text):
        bow = self._dictionary.doc2bow(text);
        tfidf = self._model[bow];
        return list(matutils.corpus2dense([tfidf], num_terms=len(self._dictionary)).T[0]);

    def show_topics(self, num_topics=-1):
        print(self._model)

class LSIModel:
    def __init__(self, dictionary, datafiles, num_topics=4):
        self._dictionary = dictionary;
        self._num_topics = num_topics;
        self._model = self._init_model(datafiles);

    def _init_model(self, datafiles):
        corpus = [];
        for datafile in datafiles:
            corpus.extend([self._dictionary.doc2bow(line) for line in myutil.tokenize_file(datafile)]);
        tfidf = models.TfidfModel(corpus);
        return models.LsiModel(corpus=tfidf[corpus], id2word=self._dictionary, num_topics=self._num_topics);

    def to_feature_vec(self, text):
        bow = self._dictionary.doc2bow(text);
        lsi = self._model[bow];
        return list(matutils.corpus2dense([lsi], num_terms=self._num_topics).T[0]);

    def show_topics(self, num_topics=-1):
        topics = self._model.show_topics(num_topics=num_topics);
        for i in range(len(topics)):
            print("#%d: %s" % (i, topics[i]))

class LDAModel:

    def __init__(self, dictionary, datafiles, num_topics=4):
        self._dictionary = dictionary;
        self._num_topics = num_topics;
        self._model = self._init_model(datafiles);

    def _init_model(self, datafiles):
        corpus = [];
        for datafile in datafiles:
            corpus.extend([self._dictionary.doc2bow(line) for line in myutil.tokenize_file(datafile)]);
        tfidf = models.TfidfModel(corpus);
        return models.LdaModel(corpus=tfidf[corpus], id2word=self._dictionary, num_topics=self._num_topics);

    def to_feature_vec(self, text):
        bow = self._dictionary.doc2bow(text);
        lsa = self._model[bow];
        return list(matutils.corpus2dense([lsa], num_terms=self._num_topics).T[0]);

    def show_topics(self, num_topics=-1):
        topics = self._model.show_topics(num_topics=num_topics);
        for i in range(len(topics)):
            print("#%d: %s" % (i, topics[i]))

'''
 リスト値を持つマップのキーを、値の要素数が多い順に並べて返します。
'''
def sortByAmount(map):
    return [keys[0] for keys in sorted(map.items(), key=lambda x: -len(x[1]))];

if (__name__ == "__main__"):

    # 引数指定
    optParser = OptionParser();
    optParser.add_option("-i", dest="infile", default="data/it1.txt");
    optParser.add_option("-o", dest="outfile", default="work/out.tsv");
    optParser.add_option("-c", dest="num_categories", default="2");
    optParser.add_option("-d", dest="dict_file", default="work/dictionary");
    (options, args) = optParser.parse_args();
    num_categories = int(options.num_categories);
    print("num_categories=%d, dictionary=%s, infile=%s, outfile=%s" % (num_categories, options.dict_file, options.infile, options.outfile));
    
    # ワード辞書
    dictionary = corpora.Dictionary.load_from_text(options.dict_file);
    # 学習/評価用データセットリスト
    datasets = [options.infile]
    model = TFIDFModel(dictionary, datasets)

    feature_vec_list = {}
    tokens_line = []
    for (tokens, line) in myutil.tokenize_file("data/it1.txt", include_line=True):
        tokens_line.extend(tokens)
    feature_vec_list.update({"it1":model.to_feature_vec(tokens_line)})

    tokens_line = []
    for (tokens, line) in myutil.tokenize_file("data/it2.txt", include_line=True):
        tokens_line.extend(tokens)
    feature_vec_list.update({"it2":model.to_feature_vec(tokens_line)})

    tokens_line = []
    for (tokens, line) in myutil.tokenize_file("data/it3.txt", include_line=True):
        tokens_line.extend(tokens)
    feature_vec_list.update({"it3":model.to_feature_vec(tokens_line)})

    tokens_line = []
    for (tokens, line) in myutil.tokenize_file("data/it4.txt", include_line=True):
        tokens_line.extend(tokens)
    feature_vec_list.update({"it4":model.to_feature_vec(tokens_line)})

    tokens_line = []
    for (tokens, line) in myutil.tokenize_file("data/it5.txt", include_line=True):
        tokens_line.extend(tokens)
    feature_vec_list.update({"it5":model.to_feature_vec(tokens_line)})

    tokens_line = []
    for (tokens, line) in myutil.tokenize_file("data/dokujo1.txt", include_line=True):
        tokens_line.extend(tokens)
    feature_vec_list.update({"dokujo1":model.to_feature_vec(tokens_line)})

    tokens_line = []
    for (tokens, line) in myutil.tokenize_file("data/dokujo2.txt", include_line=True):
        tokens_line.extend(tokens)
    feature_vec_list.update({"dokujo2":model.to_feature_vec(tokens_line)})

    tokens_line = []
    for (tokens, line) in myutil.tokenize_file("data/dokujo3.txt", include_line=True):
        tokens_line.extend(tokens)
    feature_vec_list.update({"dokujo3":model.to_feature_vec(tokens_line)})

    tokens_line = []
    for (tokens, line) in myutil.tokenize_file("data/dokujo4.txt", include_line=True):
        tokens_line.extend(tokens)
    feature_vec_list.update({"dokujo4":model.to_feature_vec(tokens_line)})

    tokens_line = []
    for (tokens, line) in myutil.tokenize_file("data/dokujo5.txt", include_line=True):
        tokens_line.extend(tokens)
    feature_vec_list.update({"dokujo5":model.to_feature_vec(tokens_line)})

    print(feature_vec_list["it1"])
    print(feature_vec_list["dokujo1"])
    
    from sklearn import svm
    svc = svm.SVC()
    training_x = [feature_vec_list["it1"], feature_vec_list["it2"], feature_vec_list["dokujo1"], feature_vec_list["dokujo2"]]
    training_y = [0, 0, 1, 1]

    # training_xは、BOWでベクトル化した各文書のリスト
    # training_yは、文書のカテゴリのリスト（ラベル）
    svc.fit(training_x, training_y)
    print("predict !")
    print(svc.predict([feature_vec_list["dokujo3"]]))


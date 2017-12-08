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

class HDPModel:

	def __init__(self, dictionary, datafiles, num_topics=4):
		self._dictionary = dictionary;
		self._num_topics = num_topics;
		self._model = self._init_model(datafiles);

	def _init_model(self, datafiles):
		corpus = [];
		for datafile in datafiles:
			corpus.extend([self._dictionary.doc2bow(line) for line in myutil.tokenize_file(datafile)]);
		return models.LdaModel(corpus=corpus, id2word=self._dictionary, num_topics=self._num_topics);

	def to_feature_vec(self, text):
		bow = self._dictionary.doc2bow(text);
		hdp = self._model[bow];
		return list(matutils.corpus2dense([hdp], num_terms=self._num_topics).T[0]);

	def show_topics(self, num_topics=-1):
		topics = self._model.show_topics(num_topics=num_topics);
		for i in range(len(topics)):
			print("#%d: %s" % (i, topics[i]))

'''
 階層クラスタリングでデータを分類します。
 @param datafile 学習用データファイルのリスト
 @param model 特徴量抽出モデル
 @param num_disp 画面表示サンプル数
'''
def classify_hcluster(datafiles, model, num_disp=-1):
	feature_vecs = [];
	lines = [];
	for datafile in datafiles:
		for (tokens, line) in myutil.tokenize_file(datafile, include_line=True):
			feature_vec = model.to_feature_vec(tokens);
			feature_vecs.append(feature_vec);
			lines.append(line.decode("utf-8"));
	result = linkage(feature_vecs[0:num_disp], metric = "chebyshev", method = "average");
	#print result;
	dendrogram(result, labels=lines[0:num_disp]);
	show();

'''
 K-meansでデータを分類します。
 @param datafile 学習用データファイルのリスト
 @param model 特徴量抽出モデル
 @param map 分類結果を書き出すマップ
 @param num_categories 分類カテゴリ数
'''
def classify_kmeans(datafiles, model, map, num_categories):
	feature_vecs = [];
	lines = [];
	for datafile in datafiles:
		for (tokens, line) in myutil.tokenize_file(datafile, include_line=True):
			feature_vec = model.to_feature_vec(tokens);
			feature_vecs.append(feature_vec);
			lines.append(line);
	features = np.array(feature_vecs);
	kmeans_model = KMeans(n_clusters=num_categories, random_state=10).fit(features);
	labels = kmeans_model.labels_;
	for label, line in zip(labels, lines):
		if (label in map):
			classified_texts = map[label];
		else:
			classified_texts = [];
			map[label] = classified_texts;
		classified_texts.append(line);

'''
 データを分類します。最大スコアの特徴を分類結果とします。
 @param datafile 学習用データファイルのリスト
 @param model 特徴量抽出モデル
 @param map 分類結果を書き出すマップ
'''
def classify_best(datafiles, model, map):
	for datafile in datafiles:
		for (tokens, line) in myutil.tokenize_file(datafile, include_line=True):
			feature_vec = model.to_feature_vec(tokens);
			category_candidate = -1;
			max = 0;
			for i in range(0, len(feature_vec)):
				if (feature_vec[i] > max):
					max = feature_vec[i];
					category_candidate = i;
			if (category_candidate in map):
				classified_texts = map[category_candidate];
			else:
				classified_texts = [];
				map[category_candidate] = classified_texts;
			classified_texts.append(line);

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
	datasets = [options.infile];
	
	# 特徴抽出モデル
	#model = HDPModel(dictionary, datasets, num_categories);
	#model = LDAModel(dictionary, datasets, num_categories);
	# model = LSIModel(dictionary, datasets, num_categories);
	model = TFIDFModel(dictionary, datasets)
	# model = BoWModel(dictionary)

	# 特徴量(またはトピック内容)の表示
	model.show_topics()
	tokens = myutil.tokenize("ジョブズが最新の携帯モデルを発表する")

	print(len(model.to_feature_vec(tokens)))
	print(model.to_feature_vec(tokens))

	tokens_line = []
	for (tokens, line) in myutil.tokenize_file("data/it1.txt", include_line=True):
		tokens_line.extend(tokens)

	print(len(model.to_feature_vec(tokens_line)))
	print(model.to_feature_vec(tokens_line))





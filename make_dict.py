#coding:utf-8
#!/usr/local/bin/python

import sys;
import MeCab;
from gensim import corpora;
from optparse import OptionParser;
import re, pprint;
import myutil;

'''
 データファイルを解析してディクショナリーに変換します。
 @param datafile データファイル
 @param savefile ディクショナリー保存先ファイル
 @param filter.min_count 登場がこの回数に満たない単語は無視
 @param filter.max_rate 登場がこの割合を超える単語は無視
 @param filter.target_cls 対象語とする品詞リスト
 @param filter.stop_words 辞書から除外する単語リスト
 @param filter.min_length 対象語とする最小の単語長
'''
def make_dictionary(datafile, savefile, filter={}):
	dictionary = corpora.Dictionary([lines for lines in myutil.tokenize_file(datafile, filter=filter)]);
	min_count = filter["min_count"] if ("min_count" in filter) else 5;
	max_rate = filter["max_rate"] if ("max_rate" in filter) else 0.3;
	dictionary.filter_extremes(no_below=min_count, no_above=max_rate);
	#print(myutil.pp(dictionary.token2id));
	print(datafile + ": " + str(len(dictionary)) + " tokens");
	dictionary.save_as_text(savefile);

'''
 エントリポイント
'''
if (__name__ == "__main__"):
	# 引数指定
	optParser = OptionParser();
	optParser.add_option("-i", dest="infile", default="data/all.txt");
	optParser.add_option("-o", dest="outfile", default="work/dictionary");
	(options, args) = optParser.parse_args();
	print("infile=%s, outfile=%s" % (options.infile, options.outfile));

	# フィルタ指定
	filter = {};
	filter["stop_words"] = ["よう", "の", "する", "いる", "ある", "ない", "なる", "れる", "できる", "こと", "もの", "てる"];
	filter["target_cls"] = ["名詞", "形容詞", "動詞"];
	filter["min_count"] = 5;
	filter["max_rate"] = 0.3;

	make_dictionary(options.infile, options.outfile, filter);

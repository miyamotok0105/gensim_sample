#coding:utf-8
import sys;
import MeCab;
import re, pprint

tagger = MeCab.Tagger();
tagger.parse("")

'''
 日本語を含む配列を出力できるようにします。
'''
def pp(obj):
	pp = pprint.PrettyPrinter(indent=4, width=160);
	str = pp.pformat(obj);
	return re.sub(r"\\u([0-9a-f]{4})", lambda x: unichr(int("0x"+x.group(1), 16)), str);

'''
 渡されたノードの単語が対象語である場合、辞書に追加すべき語を返します。
 @param node ノード
 @param filter.target_cls 対象語とする品詞リスト
 @param filter.stop_words 辞書から除外する単語リスト
 @param filter.min_length 対象語とする最小の単語長
'''
def _to_target_word(node, filter={}):
	target_cls = filter["target_cls"] if ("target_cls" in filter) else [];
	stop_words = filter["stop_words"] if ("stop_words" in filter) else [];
	min_length = filter["min_length"] if ("min_length" in filter) else 2;
	text = node.surface.lower();
	attrs = node.feature.split(",");
	if (attrs[0] == "動詞"):
		text = attrs[6]; # 基本形
	if (text.isdigit()):
		return None;
	if (len(text) < min_length):
		return None;
	if (text in stop_words):
		return None;
	if (len(target_cls) > 0 and not(attrs[0] in target_cls)):
		return None;
	return text;

'''
 渡された行を形態素に分解し、対象語のみ取り出すイテレータを返します。
 @param line 分解する行
 @param filter.target_cls 対象語とする品詞リスト
 @param filter.stop_words 辞書から除外する単語リスト
 @param filter.min_length 対象語とする最小の単語長
'''
def tokenize(line, filter={}):
	node = tagger.parseToNode(line);
	while node:
		word = _to_target_word(node, filter);
		if (word != None):
			yield word;
		node = node.next;

'''
 渡されたファイルを行ごとに形態素解析するイテレータを返します。
 @param filename ファイル名
 @param include_line 結果に行オブジェクトを渡すかどうか
 @param filter.target_cls 対象語とする品詞リスト
 @param filter.stop_words 辞書から除外する単語リスト
 @param filter.min_length 対象語とする最小の単語長
'''
def tokenize_file(filename, include_line=False, filter={}):
	file = open(filename, "r");
	for line in file:
		if (include_line):
			yield ([token for token in tokenize(line, filter)], line);
		else:
			yield [token for token in tokenize(line, filter)];
	file.close();

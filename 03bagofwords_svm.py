# -*- coding: utf-8 -*-
import os
import MeCab
from gensim import corpora, models, similarities, matutils

MODELS_DIR = "data/models"

dokujo1 = "もうすぐジューン・ブライドと呼ばれる６月。独女の中には自分の式はまだなのに呼ばれてばかり……という「お祝い貧乏」状態の人も多いのではないだろうか？　さらに出席回数を重ねていくと、こんなお願いごとをされることも少なくない。"
dokujo2 = "「お願いがあるんだけど……友人代表のスピーチ、やってくれないかな？」"
dokujo3 = "「一晩で3人位の人が添削してくれましたよ。ちなみに自分以外にもそういう人はたくさんいて、その相談サイトには同じように添削をお願いする投稿がいっぱいありました」（由利さん）"
dokujo4 = "しかし「事前にお願いされるスピーチなら準備ができるしまだいいですよ。一番嫌なのは何といってもサプライズスピーチ！」と語るのは昨年だけで10万以上お祝いにかかったというお祝い貧乏独女の薫さん（35歳）"
dokujo5 = "サプライズスピーチのメリットとしては、準備していない状態なので、フランクな本音をしゃべってもらえるという楽しさがあるようだ。しかしそれも上手に対応できる人ならいいが"

itlife1 = "インテル SSD 520をMacに装着！旧式Macはどれほど高速化するのか (上)"
itlife2 = "ThinkPad X1 Hybridは使用するCPUがx86(インテルCore iなど)"
itlife3 = "初期費用、更新費用ともに無料！ジャストシステム、ヤモリが目印のセキュリティソフト"
itlife4 = "現在では、多くのユーザーがパソコンにセキュリティソフトを導入しているが、その過半数は毎年5,000円程度かかる更新費用やその手続きについて不満を持っている。"
itlife5 = "NECは2012年2月14日、個人向けデスクトップパソコン「VALUESTAR」シリーズ3タイプ16モデルを2月16日より販売すると発表した。新商品では、よりパワフルになった録画機能に加え"

class Tokenizer():
    def __init__(self):
        self.words_list = []

    def get_words_list(self):
        return self.words_list

    def _make_words(self, text):
        words = []
        m = MeCab.Tagger("-Ochasen")
        m.parseToNode('')
        node = m.parseToNode(text)
        while node:
            feature = node.feature.split(',')
            word = feature[-3] # 基本形
            if word == '*':
                word = node.surface
            if feature[0] in ['名詞', '動詞', '形容詞']  and feature[1] != '数' and feature[-1] != 'ignore':
                words.append(word)
            node = node.next
        return words

    def _append_words_list(self, words):
        # words_list = []
        self.words_list.append(words)
        return self.words_list



tk = Tokenizer()
word = tk._make_words(dokujo1)
tk._append_words_list(word)
word = tk._make_words(dokujo2)
tk._append_words_list(word)
word = tk._make_words(dokujo3)
tk._append_words_list(word)
word = tk._make_words(dokujo4)
tk._append_words_list(word)

word = tk._make_words(itlife1)
tk._append_words_list(word)
word = tk._make_words(itlife2)
tk._append_words_list(word)
word = tk._make_words(itlife3)
tk._append_words_list(word)
word = tk._make_words(itlife4)
tk._append_words_list(word)


print(tk.get_words_list())

dictionary = corpora.Dictionary(tk.get_words_list())
# dictionary.filter_extremes(no_below, no_above)
dictionary.compactify()
corpus = [dictionary.doc2bow(words) for words in tk.get_words_list()]

print(list(matutils.corpus2dense([corpus], num_terms=len(dictionary)).T[0]))

# def make_dict(words_list):
#     dictionary = corpora.Dictionary(words_list)
#     # dictionary.filter_extremes(no_below, no_above)
#     dictionary.compactify()
#     # dictionary.save(os.path.join(MODELS_DIR, "02sample.dict"))
#     return dictionary

# def make_corpus(words_list):
#     corpus = [dictionary.doc2bow(words) for words in words_list]
#     # corpora.MmCorpus.serialize(output, corpus)
#     return corpus



# tfidf_model = models.TfidfModel(corpus)
# tfidf_corpus = tfidf_model[corpus]

# #単語IDと頻度や重みのタプル
# print(tfidf_corpus)


# from sklearn import svm
# svc = svm.SVC()
# # training_xは、BOWでベクトル化した各文書のリスト
# # training_yは、文書のカテゴリのリスト（ラベル）
# svc.fit(training_x, training_y)






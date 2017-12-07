# -*- coding: utf-8 -*-

#error

import os
import MeCab
from gensim import corpora, models, similarities

#init
MODELS_DIR = "data/models"

#make_words
words = []
m = MeCab.Tagger("-Ochasen")
m.parseToNode('')
node = m.parseToNode("例によって何のイベントかは明示しておらず、「Steve Jobs Theaterでの初イベント」とのみ書いてあるが、新ホールのこけら落としを兼ねた次期iPhoneなどの発表イベントとみられる。人気ユーチューバーのMKBHDことマルケス・ブラウンリー氏がツイートした招待状の画像によると、メディア各社に送られた招待状には「Let's meet at our place」とあるだけだ。Appleは既に新キャンパスへの移転を4月から段階的に始めている。新ホールは同社の共同創業者、故スティーブ・ジョブズ氏の名を冠し、Apple Park内で最も高い丘の上からキャンパス内の緑地とメインの建物を見下ろす。1000人収容する。")
while node:
    feature = node.feature.split(',')
    word = feature[-3] # 基本形
    if word == '*':
        word = node.surface
    if feature[0] in ['名詞', '動詞', '形容詞']  and feature[1] != '数' and feature[-1] != 'ignore':
        words.append(word)
    node = node.next

print(words)
print("===============")

#make_words_list
words_list = []
words_list.append(words)

#make_dict
dictionary = corpora.Dictionary(words_list)
# dictionary.filter_extremes(no_below, no_above)
dictionary.compactify()
# dictionary.save(os.path.join(MODELS_DIR, "02sample.dict"))

#make_corpus
corpus = [dictionary.doc2bow(words) for words in words_list]
# corpora.MmCorpus.serialize(output, corpus)

#LSI
lsi = models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=2, power_iters=2)
print(len(lsi.print_topics(2)))
print(lsi.print_topics(2)[0])
print(lsi.print_topics(2)[1])
print("===============")

doc = "イベント"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]
# print(len(vec_lsi))
# print("lsi[corpus]", lsi[corpus])
index = similarities.MatrixSimilarity(lsi[corpus])
# print("vec_lsi", vec_lsi)
# print("index",index)

try:
    sims = index[vec_lsi]
    print(list(enumerate(sims)))
except:
    print("error")


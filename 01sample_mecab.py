# -*- coding: utf-8 -*-
import sys
import MeCab
#parseとparseToNodeでレスポンスが違う

#1.parse
m = MeCab.Tagger("-Ochasen")
print(m.parse("私はローラです。"))

#2.parseToNode
words = []
m.parseToNode('')
node = m.parseToNode("私はローラです。")
while node:
    feature = node.feature.split(',')
    word = feature[-3]
    words.append(word)
    node = node.next

print(words)


#-----------------------------------------

# stoplist = []
# words = []
# m.parseToNode('')
# node = m.parseToNode("私はローラです。")
# print(node)

# while node:
#     feature = node.feature.split(',')
#     word = feature[-3] # 基本形
#     if word == '*':
#         word = node.surface
#     # if feature[0] in ['名詞', '動詞', '形容詞']  and feature[1] != '数' and feature[-1] != 'ignore':
#     if feature[-1] != 'ignore':
#         if word not in stoplist:
#             words.append(word)
#     node = node.next

# print(words)



# gensim_sample

rondhuit datasetよりニュースコーパスをダウンロード。    
https://www.rondhuit.com/download.html    

- 4カテゴリ分類

    python livedoor_corpus.py

svm結果    
0.818181818182 %    
ランダムフォレスト結果    
0.909090909091 %    


メモ    

    01:mecab
    02:gensim corpora
    03:gensim bag-of-words + scikitlearn svm
    04:gensim tf-idf + scikitlearn svm
    04:gensim tf-idf + scikitlearn randam forrest
    05:gensim LSI
    06:gensim LDA
    07:gensim doc2vec


# 参考

https://github.com/yasunori/Random-Forest-Example/blob/master/    

supervised LDA    
→マージ待ち    
https://github.com/RaRe-Technologies/gensim/issues/121    
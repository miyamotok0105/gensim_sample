# gensim_sample

rondhuit datasetよりニュースコーパスをダウンロード。    
https://www.rondhuit.com/download.html    

- 4カテゴリ分類(学習)

    python livedoor_corpus.py

- 4カテゴリ分類(推定)

    python livedoor_corpus_svm_predict.py
    python livedoor_corpus_forrest_predict.py


svm結果    
0.818181818182 %   
0.818181818182 %     
0.818181818182 %    
1.6438319683074951[sec]    
1.4903700351715088[sec]    
1.4700028896331787[sec]    

ランダムフォレスト結果    
0.909090909091 %    
0.909090909091 %    
0.818181818182 %    
1.1397030353546143[sec]    
1.4087700843811035[sec]    
1.2580029964447021[sec]    

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


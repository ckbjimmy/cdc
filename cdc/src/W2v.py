'''
The MIT License (MIT)

   Copyright (c) 2016 Wei-Hung Weng

   Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 Title   : Clinical Document Classification Pipeline: W2v module
 Author  : Wei-Hung Weng
 Created : 11/26/2015
'''

#!/usr/bin/python
import re
import pandas as pd
import nltk.data
from nltk.corpus import stopwords
from nltk.stem import *
from bs4 import BeautifulSoup
from gensim.models import word2vec


def GetW2vMatrix(model, dimension):
    features = model.vocab.keys()
    vectors = []
    for i in features:
        vectors = model[features]
    featureVector = zip(features, vectors)
    columns = ["dimension" + str(i) for i in range(1, dimension+1)]
    mat = pd.DataFrame(data=vectors, index=features, columns=columns)
    return mat


def Tokenization(data, concept, stem, removeStopwords):
    if concept == False:
        data = BeautifulSoup(data).get_text()
        data = re.sub("\r\n", " ", data)
        data = re.sub("[^a-zA-Z0-9_]", " ", data)
        data = data.lower()
    if stem == True:
        stemmer = PorterStemmer()
        data = stemmer.stem(data)
    words = data.split()
    if removeStopwords == True:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words


def ToSentences(data, tokenizer, concept, stem, removeStopwords):
    data = tokenizer.tokenize(data.strip())
    sentences = []
    for s in data:
        if len(s) > 0:
            sentences.append(Tokenization(s, concept, stem, removeStopwords))
    return sentences


def W2v(data, dimension=300, window=5, subsample=1e-5, concept=False, stem=False, removeStopwords=False):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = [] 
    for item in data:
        sentences += ToSentences(item, tokenizer, concept, stem, removeStopwords) 
    print "w2v model training"
    model = word2vec.Word2Vec(sentences, workers=4, size=dimension, min_count=1, \
        window=window, sample=subsample, sg=1, batch_words=1000)
    model.save_word2vec_format("w2v.mdl", binary=True)
    if concept == False:
         model.build_vocab(sentences)
    for epoch in range(25):
        model.train(sentences)
        model.alpha -= 0.001
        model.min_alpha = model.alpha
        #featureMatrix = GetW2vMatrix(model, dimension)
        featureMatrix = []
    return model, featureMatrix

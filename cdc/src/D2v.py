#!/usr/bin/python
# -*- coding: utf-8 -*-

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

 Title   : Clinical Document Classification Pipeline: D2v module
 Author  : Wei-Hung Weng
 Created : 10/21/2016
'''

import re
import pandas as pd
import nltk.data
from nltk.corpus import stopwords
from nltk.stem import *
from bs4 import BeautifulSoup
from gensim.models import Doc2Vec

                    
def GetD2vMatrix(model, dimension):
    features = model.docvecs.doctags.keys()
    vectors = []
    for i in model.docvecs:
        vectors = model.docvecs[features]
    featureVector = zip(features, vectors)
    columns = ["dimension" + str(i) for i in range(1, dimension+1)]
    mat = pd.DataFrame(data=vectors, index=features, columns=columns)
    return mat


def D2vLabel(data, fnames, concept, stem, removeStopwords):
    from gensim.models.doc2vec import LabeledSentence
    sentenceLabel = []
    i = 0
    for item in data:
        sentenceLabel.append(LabeledSentence(words=Tokenization(item, concept, stem, removeStopwords), tags=[fnames[i]]))
        i += 1
    return sentenceLabel


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


def D2v(data, fnames, dimension=300, window=10, subsample=1e-5, concept=False, stem=False, removeStopwords=False):
    labeledSentence = D2vLabel(data, fnames, concept, stem, removeStopwords)
    model = Doc2Vec(workers=4, min_count=1, size=dimension, window=window, sample=subsample, batch_words=1000)
    model.build_vocab(labeledSentence)    
    for epoch in range(25):
        model.train(labeledSentence)
        model.alpha -= 0.001
        model.min_alpha = model.alpha
        featureMatrix = GetD2vMatrix(model, dimension)
    return model, featureMatrix
    
#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
The MIT License (MIT)

Copyright (c) 2017 Wei-Hung Weng

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

Title   : Clinical Document Classification Pipeline: Paragraph vector
Author  : Wei-Hung Weng
Created : 08/15/2017
Usage:  : python Doc2vec.py [text_path] [label_path]
'''

import sys
import os
import re
import string
import logging
import numpy as np
import pandas as pd
from random import shuffle
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
import collections
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    
def normalize_text(texts):
    texts = [x.lower() for x in texts]
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    texts = [' '.join([word for word in x.split() if word not in set(stopwords.words("english"))]) for x in texts]
    texts = [' '.join(x.split()) for x in texts]
    return(texts)


def build_dictionary(sentences, vocabulary_size):
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]
    count = [['RARE', -1]]    
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))    
    word_dict = {}
    for word, word_count in count:
        word_dict[word] = len(word_dict)
    return(word_dict)
    

def text_to_texts(sentences):
    data = []
    for sentence in sentences:
        sentence_data = []
        for word in sentence.split():
            sentence_data.append(word.lower())
        data.append(sentence_data)
    return(data)
    

def multiclass_roc_auc_score(y_true, y_pred, average='macro'):
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_true, y_pred, average="weighted")
    

def main():

    f_path = sys.argv[1]
    l_path = sys.argv[2]
    
    global vocabulary_size
    global word_dictionary
    global word_dictionary_rev
    
    vocabulary_size = 1000000
    
    print ''
    print "--- Loading data ---"
    texts = pd.read_csv(f_path, sep='\t', encoding='latin-1')
    texts = texts.ix[:, 1].values.tolist()
    label = pd.read_csv(l_path, sep='\t', header=None, encoding='latin-1').values.tolist()
    target = label
    print "Read %d rows of data" % len(texts)
    print "Read %d rows of label" % len(label)
    
    print ''
    print "--- Text processing ---"
    texts = normalize_text(texts)
    word_dictionary = build_dictionary(texts, vocabulary_size)
    word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
        
    text_data = text_to_texts(texts)
        
    train_indices = np.random.choice(len(text_data), int(round(0.7*len(text_data))), replace=False)
    test_indices = np.array(list(set(range(len(text_data))) - set(train_indices)))
    texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
    texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
    target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
    target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])
    # tokenize
    text_data_train = np.array(text_to_texts(texts_train))
    text_data_test = np.array(text_to_texts(texts_test))
    
    # paragraph vector
    print ''
    print "--- Build paragraph vector model ---"
    tagged_doc = [TaggedDocument(words=i, tags=[str(index)]) for index, i in enumerate(text_data)]
    
    model = Doc2Vec(dm=1,
                    hs=1,
                    size=600,
                    window=10,
                    min_count=1,
                    workers=4,
                    sample=1e-5,
                    iter=20,
                    dbow_words=1)
    model.build_vocab(tagged_doc)
    
    print ''
    print "--- Training ---"
    for epoch in range(20):
        print "doc2vec dm epoch " + str(epoch)
        shuffle(tagged_doc)    
        model.train(tagged_doc, total_examples=model.corpus_count, epochs=model.iter)
        model.alpha -= 0.001
        model.min_alpha = model.alpha
    
    print ''
    print "--- Save model ---"
    model.save('model.d2v')
    model.save_word2vec_format('model.w2v')
            
    # get document key / value
    print ''
    print "--- Representation performance ---"
    key = model.docvecs.doctags.keys()
    value = [i for i in model.docvecs]
    X = pd.DataFrame(value)
    X['strlabel'] = [target[int(i)][0] for i in key]
    y = X['strlabel']
    del X['strlabel']

    alg_list = [\
        LogisticRegression(penalty='l1', multi_class='ovr', class_weight=None, n_jobs=-1), \
        CalibratedClassifierCV(base_estimator=LinearSVC(penalty='l2', loss='squared_hinge', C=1.0, multi_class='ovr', class_weight=None, random_state=0, max_iter=1000), cv=5), \
        ]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        
    for a in alg_list:
        clf = OneVsRestClassifier(a).fit(X_train, y_train)
        y_pred_prob = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        pr, re, f1, xx = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        auc = multiclass_roc_auc_score(y_test, y_pred, average='weighted')
        print acc, pr, re, f1, auc


if __name__ == "__main__":
    main()
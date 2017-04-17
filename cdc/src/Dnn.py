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

Title   : DNN pipeline
Author  : Wei-Hung Weng
Created : 11/26/2016
Comment : 
Modified: 
    09/16/2016: 
        remove cnn
        csail: rpdr sg15
    09/17/2016:
        class balance
'''

#!/usr/bin/python
import os, sys, time, re, codecs, logging
import numpy as np
import scipy as sp
import pandas as pd
import cPickle as pickle 
from W2v import *

from nltk.tokenize import sent_tokenize
from gensim.models import word2vec
from gensim.corpora.dictionary import Dictionary

from sklearn.preprocessing import LabelEncoder, LabelBinarizer, label_binarize
from sklearn.cross_validation import train_test_split, StratifiedKFold
from keras.utils.np_utils import to_categorical
    
import multiprocessing
from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report

from keras import backend as K


def multiclass_roc_auc_score(truth, pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)
    return roc_auc_score(truth, pred, average=average)
    
    
def create_dictionaries(data, model, feature):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.vocab.keys(), allow_update=True)
    w2idx = {v: k+1 for k, v in gensim_dict.items()}
    w2idxl = {v.lower(): k+1 for k, v in gensim_dict.items()}
    #w2vec = {word: model[word.lower()] for word in w2idx.keys()}
    w2vec = {}
    for word in w2idx.keys():
        if feature == 'bow':
            try:
                w2vec[word.lower()] = model[word]
            except KeyError:
                w2vec[word.lower()] = [0] * model.vector_size
        else:
            try:
                w2vec[word] = model[word]
            except KeyError:
                w2vec[word] = [0] * model.vector_size
        
    def parse_dataset(data, feature):
        for key in data.keys():
            if feature == 'bow':
                txt = data[key].lower().replace('\n', '').split()
            else:
                txt = data[key].replace('\n', '').split()
            new_txt = []
            for word in txt:
                try:
                    if feature == 'bow':
                        new_txt.append(w2idxl[word])
                    else:
                        new_txt.append(w2idx[word])
                except:
                    new_txt.append(0)
            data[key] = new_txt
        return data
        
    out = parse_dataset(data, feature)
    return w2idx, w2vec, out
    
    
def Dnn(df, y, encoder, class_wt, trained_w2v, batch_size, n_epoch, \
        best_score, feature, weighting, algorithm, repeat, kfold):
    
    y = dict(zip(df.fname, y))

    print("Class weight")
    class_weight = {}
    for i in xrange(len(encoder.classes_)):
        if class_wt:
            class_weight[i] = len(y) * 1. / sum(x == i for x in y.values())
        else:
            class_weight[i] = 1
    
    print 'Read corpus for word2vec'
    t = time.time()
    raw = pd.DataFrame(df[feature]).values.tolist()
    fnames = df['fname'].values.tolist()
    print "Read %d corpus of documents" % len(raw)
    t_e = time.time() - t
    print 'Corpus reading time ' + str(t_e) + ' sec'
    
    dict_content = zip(fnames, raw)
    tok_dict = dict(dict_content)
    del raw, fnames, dict_content
    
    w2vTraining = [item for l in tok_dict.values() for item in l]
    
    print 'Load word2vec model'
    if trained_w2v == 'no':
        t_w = time.time()
        model, featureMatrix = W2v(data=w2vTraining, \
            dimension=200, window=5, subsample=1e-5, \
            stem=False, concept=True, removeStopwords=True)
        t_w2v = time.time() - t_w
        print 'Word2vec training time ' + str(t_w2v) + ' sec'
    elif trained_w2v == 'umls':
        model = word2vec.Word2Vec.load_word2vec_format('/Users/weng/_hms_phi/DeVine_etal_200.txt', binary=False)
    elif trained_w2v == 'pubmed':
        model = word2vec.Word2Vec.load_word2vec_format('/Users/weng/_hms_phi/wikipedia-pubmed-and-PMC-w2v.bin', binary=True)
    elif trained_w2v == 'google':
        model = word2vec.Word2Vec.load_word2vec_format('.bin', binary=True)        
    
    w2v_dict = {}
    for k in tok_dict.keys():
        w2v_dict[k] = tok_dict[k][0]    
    w2v_idx, w2v_vec, w2v_emb = create_dictionaries(data=w2v_dict, model=model, feature=feature)    
    
    vocab_dim = model.vector_size
    n_symbols = len(w2v_idx)

    embedding_weights = np.zeros((n_symbols+1, vocab_dim))
    for word, index in w2v_idx.items():
        try:
            embedding_weights[index, :] = w2v_vec[word]
        except KeyError:
            embedding_weights[index, :] = [0] * model.vector_size
    #embedding_weights = embedding_weights[1:]
    max_length = max((len(v), k) for k,v in w2v_emb.iteritems())[0]

    X = w2v_dict
     
    #https://github.com/fchollet/keras/issues/853
    #There are 3 approaches:
    #
    #Learn embedding from scratch - simply add an Embedding layer to your model
    #Fine tune learned embeddings - this involves setting word2vec / GloVe vectors as your Embedding layer's weights.
    #Use word word2vec / Glove word vectors as inputs to your model, instead of one-hot encoding.
    #The third one is the best option(Assuming the word vectors were obtained from the same domain as the inputs to your models. For e.g, if you are doing sentiment analysis on tweets, you should use GloVe vectors trained on tweets).
    #
    #In the first option, everything has to be learned from scratch. You dont need it unless you have a rare scenario. The second one is good, but, your model will be unnecessarily big with all the word vectors for words that are not frequently used.

    np.random.seed(777)

    print("Constrcut the dataset")
    tmp = [X, y]
    d = {}
    for k in X.iterkeys():
        d[k] = tuple(d[k] for d in tmp)
    
    X = []
    y = []
    for k, v in d.iteritems():
        X.append(v[0])
        y.append(v[1])
        
    print("Split the dataset")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=max_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_length)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    
    print("Convert labels to Numpy Sets")
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print("Convert to categories")
    y_train_dummy = to_categorical(y_train)
    y_test_dummy = to_categorical(y_test)
    
    print("Modeling")
    input_length = max_length
    cpu_count = multiprocessing.cpu_count()
    
    pred_table = pd.DataFrame()
    score_table = pd.DataFrame()
    best_score = 0
    best_model = best_feature = best_algorithm = coef = ''
    
    skf = StratifiedKFold(y=y_train, n_folds=kfold, shuffle=True, random_state=None)
    
    for idx, (train, test) in enumerate(skf):
        model = Sequential()
        model.add(Embedding(input_dim  = n_symbols+1,
                            output_dim = vocab_dim,
                            #mask_zero  = False, # need to be zero otherwise cnn won't work
                            weights    = [embedding_weights],
                            input_length = input_length,
                            trainable = True))
                            
        model.add(Convolution1D(nb_filter=64, filter_length=2, border_mode='valid', activation='relu'))
        #model.add(Convolution1D(nb_filter=32, filter_length=2, border_mode='valid', activation='relu'))
        model.add(MaxPooling1D(pool_length=2))
        model.add(Dropout(0.5))
        #model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='valid', activation='relu'))
        #model.add(MaxPooling1D(pool_length=2))
        #model.add(Dropout(0.5))
        #model.add(Bidirectional(LSTM(32)))
        #model.add(Dropout(0.5))
        model.add(Flatten())
        
        model.add(Dense(y_train_dummy.shape[1] * 10, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(y_train_dummy.shape[1], activation = 'softmax'))
        #model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        
        print("Training")
        early_stopping = EarlyStopping(monitor='val_loss', patience=3) 
        t = time.time()
        model.fit(X_train, y_train_dummy, batch_size=batch_size, nb_epoch=n_epoch, class_weight=class_weight, \
            validation_data=(X_test, y_test_dummy), shuffle = True, callbacks=[early_stopping])   
        time.sleep(0.1)     
        t_e = time.time() - t
            
        print "Model: " + feature + ' | ' + str(algorithm) + ' | rep' + str(repeat+1) + ' | cv' + str(kfold+1) + ' | time: ' + str(t_e) + ' sec'
    
        y_pred_prob = model.predict_proba(X_test)
        try:
            y_pred = model.predict_classes(X_test) # different
        except TypeError:
            y_pred = model.predict(X_test.values)
    
        t = pd.concat([pd.DataFrame(y_test), pd.DataFrame(y_pred)], axis=1)
        t.columns = ['true', 'pred']
        pred_tbl = pd.DataFrame(y_pred_prob)
        pred_tbl = pd.concat([t, pred_tbl], axis=1)
        pred_tbl['rep'] = repeat + 1
        pred_tbl['k'] = kfold + 1
        pred_tbl['algorithm'] = str(algorithm)
        pred_tbl['feature'] = feature
        pred_tbl['weighting'] = weighting
        
        acc = accuracy_score(y_test, y_pred)
        pr, re, f1, xx = precision_recall_fscore_support(y_test, y_pred, average='binary') # didn't use 'macro'
        auc = multiclass_roc_auc_score(y_test, y_pred, average='weighted') # weighted AUC takes imbalanced label into account
        print acc, pr, re, f1, auc
        
        metrics = pd.DataFrame([0])
        metrics['time'] = t_e
        metrics['accuracy'] = acc
        metrics['precision'] = pr
        metrics['recall'] = re
        metrics['f1'] = f1
        metrics['auc'] = auc
        metrics['rep'] = repeat + 1
        metrics['k'] = idx + 1
        metrics['algorithm'] = str(algorithm)
        metrics['feature'] = feature
        metrics['weighting'] = weighting
        
        if auc > best_score:
            best_model = model
            best_score = auc
            best_feature = feature
            best_algorithm = str(algorithm)[0:9]
            best_coef = coef
        
        #encoder.inverse_transform(y_pred) #real label name
        
        Y_test = np.argmax(y_test_dummy, axis=1)
        Y_pred = np.argmax(y_pred_prob, axis=1)
        print(classification_report(Y_test, Y_pred))
        
        pred_table = pd.concat([pred_table, pred_tbl], axis=0)
        score_table = pd.concat([score_table, metrics], axis=0)
        
    return best_model, best_feature, best_algorithm, best_coef, score_table, pred_table, metrics
        
#j = open('best_dnn_model.json', 'r')
#json = j.read()
#j.close()
#model = model_from_json(json)
#model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#score = model.evaluate(X_test, y_test_dummy, verbose=0)
    
#https://github.com/dandxy89/DeepLearning_MachineLearning/blob/master/EmbeddingKeras/imdb_embedding_w2v.py
#http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
#https://github.com/fchollet/keras/issues/1629 (consider biridectional lstm)
#http://datascience.stackexchange.com/questions/10048/what-is-the-best-keras-model-for-multi-label-classification (for binary)

#
#    '''
#    as feature
#    '''
#    print("Use layer output as features")
#    #https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
#    print("Extract layer output")
#    get_8th_layer_output = K.function([model.layers[0].input, K.learning_phase()],
#                                    [model.layers[8].output])
#    output_train = get_8th_layer_output([X_train, 1])[0]
#    output_test = get_8th_layer_output([X_test, 0])[0]
#    
#    
#    rf(output_train, y_train, output_test, y_test)
#    svc(output_train, y_train, output_test, y_test)
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

Title   : Clinical Document Classification Pipeline: NLP-Supervised learning module
Author  : Wei-Hung Weng
Created : 10/21/2016
Note    : 11/26/2016: change linear SVM kernel, add SGD, AUC
'''

import os, sys, re, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, label_binarize

import string
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from D2v import *
from Dnn import *

from sklearn.feature_selection import SelectKBest, SelectFromModel, GenericUnivariateSelect
from sklearn.feature_selection import chi2
from sklearn.linear_model import LassoCV

from joblib import Parallel, delayed
import multiprocessing

from sklearn.utils import shuffle
from sklearn.cross_validation import KFold, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier, OutputCodeClassifier
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
import cPickle


def multiclass_roc_auc_score(truth, pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)
    return roc_auc_score(truth, pred, average=average)
    
	           
def modeling(path, best_score, feature, algorithm, weighting, X_train, y_train, X_test, y_test, repeat=0, kfold=0, tm=0.):
    
    time_model = time.time()
    
    best_model = best_feature = best_algorithm = best_coef = ''
    coef = ''
    
    clf = OneVsRestClassifier(algorithm).fit(X_train, y_train)
    try:
        coef = clf.estimators_.feature_importances_
    except:
        next
    
    time_end = time.time() - time_model + tm
    
    print "Model: " + feature + ' | ' + str(algorithm) + ' | rep' + str(repeat+1) + ' | cv' + str(kfold+1) + ' | time: ' + str(time_end) + ' sec'
    
    y_pred_prob = clf.predict_proba(X_test)
    try:
        y_pred = clf.predict(X_test)
    except TypeError:
        y_pred = clf.predict(X_test.values)
    
    t = pd.concat([pd.DataFrame(y_test), pd.DataFrame(y_pred)], axis=1)
    t.columns = ['true', 'pred']
    pred_tbl = pd.DataFrame(y_pred_prob, columns=list(clf.classes_))
    pred_tbl = pd.concat([t, pred_tbl], axis=1)
    pred_tbl['rep'] = repeat + 1
    pred_tbl['k'] = kfold + 1
    pred_tbl['algorithm'] = str(algorithm)
    pred_tbl['feature'] = feature
    pred_tbl['weighting'] = weighting
    
    acc = accuracy_score(y_test, y_pred)
    pr, re, f1, xx = precision_recall_fscore_support(y_test, y_pred, average='weighted') # didn't use 'macro'
    auc = multiclass_roc_auc_score(y_test, y_pred, average='weighted') # weighted AUC takes imbalanced label into account
    print acc, pr, re, f1, auc
    
    metrics = pd.DataFrame([0])
    metrics['time'] = time_end
    metrics['accuracy'] = acc
    metrics['precision'] = pr
    metrics['recall'] = re
    metrics['f1'] = f1
    metrics['auc'] = auc
    metrics['rep'] = repeat + 1
    metrics['k'] = kfold + 1
    metrics['algorithm'] = str(algorithm)
    metrics['feature'] = feature
    metrics['weighting'] = weighting
    
    if auc > best_score:
        best_model = clf
        best_score = auc
        best_feature = feature
        best_algorithm = str(algorithm)[0:9]
        best_coef = coef
    
    return best_model, best_feature, best_algorithm, best_coef, pred_tbl, metrics
        
        
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    import re
    tokens = word_tokenize(text)
    output = [i for i in tokens if i not in string.punctuation and not re.match("^[0-9.].*$", i) and len(i) > 2]
    output = stem_tokens(output, stemmer)
    return output
    
    
def MLPipeline(path, data, label, vec, alg, cwt, feature, rep, k):
    if cwt == 'balanced':
        class_wt = cwt
    else:
        class_wt = None

    alg_abbreviation = ['l1', 'l2', 'nb', 'svmlin', 'svmsgd', 'svmrbf', 'rf', 'adaboost', 'mlp', 'dnn']
    alg_list = [\
        LogisticRegression(penalty='l1', multi_class='ovr', class_weight=class_wt, n_jobs=-1), \
        LogisticRegression(penalty='l2', multi_class='ovr', class_weight=class_wt, n_jobs=-1), \
        MultinomialNB(), \
        #SVC(kernel='linear', probability=True, decision_function_shape='ovr', class_weight=class_wt), \
        CalibratedClassifierCV(base_estimator=LinearSVC(penalty='l2', loss='squared_hinge', C=1.0, multi_class='ovr', class_weight=class_wt, random_state=0, max_iter=1000), cv=5), \
        CalibratedClassifierCV(base_estimator=SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True, epsilon=0.1, n_jobs=-1, random_state=0, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=class_wt), cv=5), \
        SVC(kernel='rbf', probability=True, decision_function_shape='ovr', class_weight=class_wt), \
        RandomForestClassifier(n_estimators=100, class_weight=class_wt, n_jobs=-1), \
        AdaBoostClassifier(n_estimators=100), \
        #GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)] # GBM can't use sparse matrix, scikit-learn problem
        MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(128, 64, 32), random_state=42), \
        'dnn']
            
    a = alg.split('+')
    alg_list = [alg_list[alg_abbreviation.index(i)] for i in a]
    num_cores = multiprocessing.cpu_count()

    os.chdir(path + 'data/')
    
    print "--- Data preprocessing ---"
    df = pd.read_csv(data, sep='\t')
    df = df.sort('fname', ascending=True).reset_index(drop=True)
    df['label'] = pd.read_csv(label, sep='\t', header=None)
    #for i in xrange(len(df.fname.tolist())):
    #    df.label[i] = re.sub(r"(.*)(.*)( \([0-9]+\).xml)", r"\1", df.fname.tolist()[i])
    y_unencoded = df.label

    print "Label encoding"
    encoder = LabelEncoder()
    encoder.fit(y_unencoded)
    y = encoder.transform(y_unencoded)

    pred_table = pd.DataFrame()
    score_table = pd.DataFrame()
    best_score = 0
    
    ff = feature.split('+')
    
    for f in ff:
        
        time_start = time.time()
        
        if alg == 'dnn':
            best_model, best_feature, best_algorithm, best_coef, score_table, pred_table, metrics = \
                Dnn(df=df, y=y, encoder=encoder, class_wt=False, trained_w2v='umls', batch_size=int(df.shape[0]/100), n_epoch=25, \
                    best_score=best_score, feature=f, weighting='w2v', algorithm='cnn', repeat=rep, kfold=k)
                
        else:
            print "--- Vector representation ---"
            
            try:
                X = pd.DataFrame(df[f])
            except KeyError:
                print "Feature combination"
                f_split = f.split('_')
                X = pd.DataFrame(df[f_split].apply(lambda x: ' '.join(x), axis=1))
                
            X.columns = ['concept']
            
            stopWords = stopwords.words('english')
            stemmer = PorterStemmer()
    
            if vec == 'freq':
                print "One-hot representation"
                if f == 'bow':
                    v = CountVectorizer(tokenizer=tokenize, stop_words=stopWords, lowercase=True)
                else:
                    v = CountVectorizer(tokenizer=tokenize, stop_words=stopWords)
                print "Convert to sparse matrix"
                X = v.fit_transform(X.concept)
    
                
            elif vec == 'tfidf':
                print "Tf-idf representation"
                if f == 'bow':
                    v = TfidfVectorizer(tokenizer=tokenize, stop_words=stopWords, lowercase=True, norm='l2', use_idf=True, \
                        smooth_idf=True, sublinear_tf=True)
                else:
                    v = TfidfVectorizer(tokenizer=tokenize, stop_words=stopWords, lowercase=True, norm='l2', use_idf=True, \
                        smooth_idf=True, sublinear_tf=True)
                print "Convert to sparse matrix"
                X = v.fit_transform(X.concept)
            
            elif vec == 'pv':
                print "Paragraph vector representation"
                if f == 'bow':
                    c, s, r = [False, True, True]
                else:
                    c, s, r = [True, False, False]
                    
                try: 
                    model, featureMatrix = D2v(data=df[f], fnames=df['fname'].values.tolist(), \
                        dimension=600, window=10, subsample=1e-5, concept=c, stem=s, removeStopwords=r)
                except KeyError:
                    d2v_input = f.split('_')
                    d2v_input = pd.DataFrame(df[d2v_input].apply(lambda x: ' '.join(x), axis=1))
                    model, featureMatrix = D2v(data=d2v_input[0], fnames=df['fname'].values.tolist(), \
                        dimension=600, window=10, subsample=1e-5, concept=c, stem=s, removeStopwords=r)
                    
                featureMatrix['order'] = pd.Categorical(featureMatrix.index, categories=df.fname.values.tolist(), ordered=True)
                X = featureMatrix.sort('order')
                del X['order']
    
            print str(X.shape)
            
            time_feature = time.time() - time_start
            
            print "Feature selection (N)"
            #http://scikit-learn.org/stable/modules/feature_selection.html
            #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV
            #http://stackoverflow.com/questions/14133348/show-feature-names-after-feature-selection
            #chi2 = GenericUnivariateSelect(chi2, mode='fdr', param=0.05).fit(X, y)
            #X_new = chi2.transform(X)
            #X_new = pd.DataFrame(X_new, columns=np.asarray(vec.get_feature_names())[chi2.get_support()])
            #
            #lasso = SelectFromModel(LassoCV(cv=10)).fit(X, y)
            #X_new = lasso.transform(X)
            #X_new = pd.DataFrame(X_new, columns=np.asarray(vec.get_feature_names())[lasso.get_support()])
            
            
            print "Topic modeling (N)"
            #http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html
            
            print "--- Supervised learning with repeated cross-validation ---"
            print "input X, y, algorithm, rep, k"
            
            alg_num = xrange(len(alg_list))
            
            for a in alg_list:
                
                for i in xrange(rep):
                    #X, y = shuffle(X, y, random_state=42)
                    
                    #kf = KFold(len(y), n_folds=k, shuffle=True, random_state=None)
                    skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=None)
                    
                    for idx, (train, test) in enumerate(skf):
                        if vec == 'freq' or vec == 'tfidf':
                            X_train, X_test = X[train], X[test]
                            y_train, y_test = y[train], y[test]
                        else: #pv
                            X_train = X.iloc[train]
                            X_test = X.iloc[test]
                            y_train = y[train]
                            y_test = y[test]
                        
                        best_model, best_feature, best_algorithm, best_coef, pred_tbl, metrics = modeling(path=path, best_score=best_score, feature=f, algorithm=a, \
                            weighting=vec, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, repeat=i, kfold=idx, tm=time_feature)                    
                        pred_table = pd.concat([pred_table, pred_tbl], axis=0)
                        score_table = pd.concat([score_table, metrics], axis=0)
    
    try:
        cPickle.dump(v.vocabulary_, open(path + 'model/feature_f=' + best_feature + '_a=' + best_algorithm + '.pkl', 'wb'))
    except:
        next  
        
    if alg == 'dnn':
        with open(path + 'model/model_f=' + best_feature + '_a=' + best_algorithm + '.json', 'wb') as json:
            json.write(best_model.to_json())
    else:
        cPickle.dump(best_model, open(path + 'model/model_f=' + best_feature + '_a=' + best_algorithm + '.pkl', 'wb'))

    cPickle.dump(v, open(path + 'model/vectorizer_f=' + best_feature + '_a=' + best_algorithm + '.pkl', 'wb'))  
    cPickle.dump(encoder, open(path + 'model/encoder_f=' + best_feature + '_a=' + best_algorithm + '.pkl', 'wb'))  
    score_table.to_csv(path + 'result/' + vec + '_' + alg + '_'+ cwt + '_' + feature + '_' + str(rep) + '_' + str(k) + '_score.txt', sep='\t')
    #score_table = pd.read_csv(path + 'result/' + vec + '_' + alg + '_'+ cwt + '_' + feature + '_' + str(rep) + '_' + str(k) + '_score.txt', sep='\t')
    pred_table.to_csv(path + 'result/' + vec + '_' + alg + '_'+ cwt + '_' + feature + '_' + str(rep) + '_' + str(k) + '_pred.txt', sep='\t')
    #pred_table = pd.read_csv(path + 'result/' + vec + '_' + alg + '_'+ cwt + '_' + feature + '_' + str(rep) + '_' + str(k) + '_pred.txt', sep='\t')
    print best_coef
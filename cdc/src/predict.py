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

Title   : Clinical Document Classification Pipeline: Predict function
Author  : Wei-Hung Weng
Created : 10/25/2016
'''

import os, sys, getopt, re
import cPickle
from psutil import virtual_memory
from NoteDeid import *
from NoteConceptParser import *
from Converter import *
from MLPipeline import *
from D2v import *


def main(argv):    
    mem = virtual_memory().total / 1024.**3
    
    datadir = parser = parserdir = model_file = ''
    
    try:
        opts, args = getopt.getopt(argv, "d:p:q:m:h", \
        ["datadir=", "parser=", "parserdir=", "model_file=", "help"])
    except getopt.GetoptError:
        usage('predict.py -d <datadir> -p <parser> -q <parserdir> -m <model_file> -h')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == ("-h", "--help"):
            usage('predict.py -d <datadir> -p <parser> -q <parserdir> -m <model_file> -h')
            sys.exit()
        elif opt in ("-d", "--datadir"):
            datadir = arg
        elif opt in ("-p", "--parser"):
            parser = arg
        elif opt in ("-q", "--parserdir"):
            parserdir = arg
        elif opt in ("-m", "--model_file"):
            model_file = arg

    print 'Data directory           :', datadir
    print 'Running concept parser   :', parser
    print 'Concept parser directory :', parserdir
    print 'Location of model        :', model_file
    
    feature_file = model_file.replace("/model_", "/feature_")
    encoder_file = model_file.replace("/model_", "/encoder_")
    feat = re.sub(r"(.*model_f=)(.*)(_a=.*)", r"\2", model_file)
    
    cwd = os.getcwd()
    os.system("mkdir cdc_tmp; mkdir cdc_tmp/data")
    workdir = cwd + "/cdc_tmp/"
    os.chdir(workdir)

    if os.path.exists(datadir) == False:
        print "Please assign the data source"
        sys.exit(2)
        
    if os.path.isfile(datadir):
        os.system("mkdir tmp_data")
        dd = workdir + "tmp_data/"
        os.system("cp " + datadir + " " + dd)
        datadir = dd
        
    #if bool(re.compile('.*model_bow_[A-Za-z]+.pkl').match(model_file)):
    if feat == "bow":
        Converter(workdir=workdir, folder=datadir, format="txt")
        df = pd.read_csv(workdir + 'data/data.txt', sep='\t')
        df = df.sort_values(by='fname', ascending=True).reset_index(drop=True)
        X = pd.DataFrame(df['bow'])
        X.columns = ['concept']
        v = CountVectorizer(tokenizer=tokenize, stop_words=stopWords, lowercase=True, \
            vocabulary=cPickle.load(open(feature_file, "rb")))
    else:
        if parser == "ctakes":
            RunCtakes(folder=datadir, ctakesDir=parserdir, erisOne=None) 
            os.system('cd xml; find . -name "*.xml" -exec mv {} ' + workdir + 'data/ \;')
            Converter(workdir=workdir, folder=workdir + 'data/', format="xml")
            df = pd.read_csv(workdir + 'data/data.txt', sep='\t')
            df = df.sort_values(by='fname', ascending=True).reset_index(drop=True)
            
            try:
                X = pd.DataFrame(df[feat])
            except KeyError:
                print "Feature combination"
                f_split = feat.split('_')
                X = pd.DataFrame(df[f_split].apply(lambda x: ' '.join(x), axis=1))
            X.columns = ['concept']
            if feat == "bow":
                v = CountVectorizer(tokenizer=tokenize, stop_words=stopWords, lowercase=True, \
                    vocabulary=cPickle.load(open(feature_file, "rb")))
            else:
                v = CountVectorizer(stop_words=stopWords, \
                    vocabulary=cPickle.load(open(feature_file, "rb")))
    
    print "Convert to sparse matrix"
    X = v.fit_transform(X.concept)
    
    clf = cPickle.load(open(model_file, 'rb'))
    v = cPickle.load(open(vec_file, 'rb'))
    y_pred_prob = clf.predict_proba(X)
    try:
        y_pred = clf.predict(X)
    except TypeError:
        y_pred = clf.predict(X.values)

    encoder = cPickle.load(open(encoder_file, 'rb'))
    result = pd.DataFrame(y_pred_prob)
    result.columns = encoder.classes_ 
    result['pred_class'] = encoder.inverse_transform(y_pred)
    result['fname'] = df['fname']
    result.to_csv(cwd + '/result.csv', sep="\t")
    print(result)
        
    os.chdir(cwd)

if __name__ == "__main__":
    main(sys.argv[1:])
    

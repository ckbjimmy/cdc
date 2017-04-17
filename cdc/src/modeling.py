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

Title   : Clinical Document Classification Pipeline: Modeling function
Author  : Wei-Hung Weng
Created : 10/21/2016
'''

import os, sys, getopt
from psutil import virtual_memory
from NoteDeid import *
from NoteConceptParser import *
from Converter import *
from MLPipeline import *
from D2v import *
from RPDR import *

def main(argv):    
    mem = virtual_memory().total / 1024.**3
    
    workdir = datadir = labelfile = deid = parser = parserdir = ctakessetup = utsname = utspw = \
    ctakesminmem = ctakesmaxmem = convert = model = feature = vec_represent = algorithm = \
    balanced_class = repeat = cv = rpdr = ''
    
    try:
        opts, args = getopt.getopt(argv, "w:d:l:i:p:q:s:n:o:e:g:c:m:f:v:a:b:r:k:x:h", \
        ["workdir=", "datadir=", "labelfile=", "deid=", "parser=", "parserdir=", "ctakessetup=", "utsname=", \
        "utspw=", "ctakesminmem=", "ctakesmaxmem=", "convert=", "model=", "feature=", "vec_represent=", \
        "algorithm=", "balancedclass=", "repeat=", "cv=", "rpdr=", "help"])
    except getopt.GetoptError:
        usage('pipeline.py -wd <workdir> -dd <datadir> -lf <labelfile> -d <T/F> -p <T/F> -pd <parserdir> -cset <T/F> -un <utsname> -upw <utspw> -mnm <ctakesminmem> -mxm <ctakesmaxmem> -c <convert> -m <T/F> -f <feature> -v <vec_represent> -a <algorithm> -bc <T/F> -r <repeat> -cv <cv> -h')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == ("-h", "--help"):
            usage('pipeline.py -wd <workdir> -dd <datadir> -lf <labelfile> -d <T/F> -p <T/F> -pd <parserdir> -cset <T/F> -un <utsname> -upw <utspw> -mnm <ctakesminmem> -mxm <ctakesmaxmem> -c <convert> -m <T/F> -f <feature> -v <vec_represent> -a <algorithm> -bc <T/F> -r <repeat> -cv <cv> -h')
            sys.exit()
        elif opt in ("-w", "--workdir"):
            workdir = arg
        elif opt in ("-d", "--datadir"):
            datadir = arg
        elif opt in ("-l", "--labelfile"):
            labelfile = arg
        elif opt in ("-i", "--deid"):
            deid = arg
        elif opt in ("-p", "--parser"):
            parser = arg
        elif opt in ("-q", "--parserdir"):
            parserdir = arg
        elif opt in ("-s", "--ctakessetup"):
            ctakessetup = arg
        elif opt in ("-n", "--utsname"):
            utsname = arg
        elif opt in ("-o", "--utspw"):
            utspw = arg
        elif opt in ("-e", "--ctakesminmem"):
            ctakesminmem = arg
        elif opt in ("-g", "--ctakesmaxmem"):
            ctakesmaxmem = arg
        elif opt in ("-c", "--convert"):
            convert = arg
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-f", "--feature"):
            feature = arg
        elif opt in ("-v", "--vec_represent"):
            vec_represent = arg
        elif opt in ("-a", "--algorithm"):
            algorithm = arg
        elif opt in ("-b", "--balancedclass"):
            balanced_class = arg
        elif opt in ("-r", "--repeat"):
            repeat = int(arg)
        elif opt in ("-k", "--cv"):
            cv = int(arg)
        elif opt in ("-x", "--rpdr"):
            xx = arg
    
    print 'Working directory        :', workdir
    print 'Data directory           :', datadir
    print 'Label file location      :', labelfile
    print 'Running deidentification :', deid
    print 'Running concept parser   :', parser
    print 'Concept parser directory :', parserdir
    print 'Setting up cTAKES        :', ctakessetup
    print 'UTS username             :', utsname
    print 'UTS password             :', utspw
    print 'cTAKES minimal memory    :', ctakesminmem
    print 'cTAKES maximal memory    :', ctakesmaxmem
    print 'Data conversion from     :', convert
    print 'Modeling                 :', model
    print 'Selected feature         :', feature
    print 'Vector representation    :', vec_represent
    print 'Algorithm                :', algorithm
    print 'Class balancing          :', balanced_class
    print 'Repeat time              :', repeat
    print 'k-fold cross-validation  :', cv

    if os.path.exists(workdir) == False:
        os.system("mkdir " + workdir + "; cd " + workdir + "; mkdir data; mkdir model; mkdir result; cd ..")
        
    os.chdir(workdir)
    
    if datadir == '' and 'xx' not in locals():
        datadir = workdir + 'data/'
        print "No data provided, use sample data"
        os.system("wget https://github.com/ckbjimmy/ckbjimmy.github.io/raw/master/docs/sample.tar.gz; \
        wget https://raw.githubusercontent.com/ckbjimmy/ckbjimmy.github.io/master/docs/label.txt; \
        tar -xzf sample.tar.gz; mv sample/ data/xml/; mv label.txt data/label.txt; rm sample.tar.gz")
        labelfile = datadir + 'label.txt'
        
    if deid == "T":
        try: 
            RunDeid(folder=datadir, deidDir=workdir + 'deid-1.1')
        except:
            print "Downloading deid package from physionet"
            os.chdir(workdir)
            os.system("wget https://www.physionet.org/physiotools/sources/deid/deid-1.1.tar.gz; \
        tar -xzf deid-1.1.tar.gz; rm deid-1.1.tar.gz; cd deid-1.1; rm *.txt; cd ..")
            try:
                RunDeid(folder=datadir, deidDir=workdir + 'deid-1.1')
            except:
                print "No free text files for deidentification"
                sys.exit(2)

    if parser == "ctakes":
        if ctakessetup == "T":
            if os.path.exists(parserdir):
                SetupCtakes(parserdir, utsname, utspw, ctakesminmem, ctakesmaxmem)
                RunCtakes(folder=datadir, ctakesDir=parserdir, erisOne=None)
            else:
                print "Please download cTAKES and assign directory"
                sys.exit(2)
        else:
            RunCtakes(folder=datadir, ctakesDir=parserdir, erisOne=None)
    
    if convert == "xml":
        try:
            Converter(workdir=workdir, folder=datadir + 'xml/', format=convert)
        except OSError:
            print "No XML output files inside the folder. Must add argument -p ctakes."
            sys.exit(2)
    elif convert == "txt" or convert == "res":
        Converter(workdir=workdir, folder=datadir, format=convert)
    elif convert == "cas":
        next
    elif convert == "rpdr":
        next

    if model == "T":
        if convert == "idash":
            MLPipeline(path=workdir, data=workdir + 'data/data.txt', label=workdir + 'data/label.txt', \
            vec=vec_represent, alg = algorithm, cwt=balanced_class, feature=feature, rep=repeat, k=cv)
        elif convert == "rpdr":
            rpdr_param = xx.split('+')
            RPDR(path=workdir, lab=rpdr_param[1], num=rpdr_param[2])
            MLPipeline(path=workdir, data=workdir + 'data/data.txt', label=workdir + 'data/label.txt', \
            vec=vec_represent, alg = algorithm, cwt=balanced_class, feature=feature, rep=repeat, k=cv)
        else:
            MLPipeline(path=workdir, data=workdir + 'data/data.txt', label=labelfile, \
            vec=vec_represent, alg = algorithm, cwt=balanced_class, feature=feature, rep=repeat, k=cv)
        
    print "Done!"

if __name__ == "__main__":
    main(sys.argv[1:])

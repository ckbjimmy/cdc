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

Title   : Clinical Document Classification Pipeline: Note concept parser module (cTAKES)
Author  : Wei-Hung Weng
Created : 10/21/2016
'''

import sys, os, time
import subprocess
import commands
import string
import pandas as pd


def SetupCtakes(parserdir, utsname, utspw, ctakesminmem, ctakesmaxmem):
    print "cTAKES setup"
    cwd = os.getcwd()
    os.chdir(parserdir + "bin/")    
    os.system("sed '45d' runctakesCPE.sh > pipeline.sh")
    os.system("echo 'java -Dctakes.umlsuser=" + utsname + " -Dctakes.umlspw=" + utspw + " -cp $CTAKES_HOME/lib/*:$CTAKES_HOME/desc/:$CTAKES_HOME/resources/ -Dlog4j.configuration=file:$CTAKES_HOME/config/log4j.xml -Xms" + ctakesminmem + " -Xmx" + ctakesmaxmem + " org.apache.uima.examples.cpe.SimpleRunCPE $CTAKES_HOME/desc/ctakes-clinical-pipeline/desc/collection_processing_engine/test_plaintext_test.xml' >> pipeline.sh")
    os.system("chmod u+x pipeline.sh")
    os.chdir(parserdir)
    os.system('mkdir note_input; mkdir note_output')
    os.chdir(parserdir + "desc/ctakes-clinical-pipeline/desc/collection_processing_engine/")
    os.system("cp test_plaintext.xml test_plaintext_test.xml")
    os.system("sed -ie 's/testdata\\/cdptest\\/testinput\\/plaintext/note_input/g' test_plaintext_test.xml")
    os.system("sed -ie 's/testdata\\/cdptest\\/testoutput\\/plaintext/note_output/g' test_plaintext_test.xml")
    os.system("sed -ie 's/AggregatePlaintextProcessor/AggregatePlaintextUMLSProcessor/g' test_plaintext_test.xml")
    os.system("rm test_plaintext_test.xmle")
    os.chdir(cwd)


def RunCtakes(folder, ctakesDir, erisOne):
    print "Running cTAKES"
    t = time.time()
    os.chdir(folder)
    try:
        subprocess.check_output(['bash', '-c', 'rm -rf .DS_Store'])
    except:
        next
    subprocess.check_output(['bash', '-c', 'find . -iname "*.txt" -exec bash -c \'cp "$0" "${0%\.txt}.res"\' {} \;'])
    subprocess.check_output(['bash', '-c', 'find . -iname "*.res" -exec bash -c \'cp "$0" "${0%\.res}.txt"\' {} \;'])
    #subprocess.check_output(['bash', '-c', 'mkdir ctakes'])
    os.system("mkdir ctakes")
    subprocess.check_output(['bash', '-c', 'cp -r ' + ctakesDir + ' ctakes/'])
    subprocess.check_output(['bash', '-c', 'find . -name "*.res" -exec mv {} ctakes/note_input/ \;'])
    try:
        #subprocess.check_output(['bash', '-c', 'rm -rf ctakes/note_input/.DS_Store'])
        #subprocess.check_output(['bash', '-c', 'rm -rf ctakes/note_input/id.res'])
        os.system("rm -rf ctakes/note_input/.DS_Store")
        os.system("rm -rf ctakes/note_input/id.res")
    except:
        next
    #subprocess.check_output(['bash', '-c', 'mkdir deid'])
    os.system("mkdir deid")
    subprocess.check_output(['bash', '-c', 'find . -name "*.text" -exec mv {} deid/ \;'])
    subprocess.check_output(['bash', '-c', 'find . -name "*.info" -exec mv {} deid/ \;'])
    subprocess.check_output(['bash', '-c', 'find . -name "*.phi" -exec mv {} deid/ \;'])
    #subprocess.check_output(['bash', '-c', 'mkdir xml'])
    os.system("mkdir xml")

    os.chdir(folder + "/ctakes/bin/")
    cmd = "sh pipeline.sh"
    if erisOne == "eris":
        with open("eris.lsf", "w") as f:
            f.write(cmd)
        eris = "bsub -q big -n 4 -R 'rusage[mem=8000]' < eris.lsf"
        subprocess.check_output(['bash', '-c', eris])
    else:
        print cmd
        print os.getcwd()
        print "cTAKES parsing"
        os.system(cmd)
        os.chdir(folder + "/ctakes/note_input/")    
        subprocess.check_output(['bash', '-c', 'find . -name "*.res" -exec mv {} ' + folder + '/deid/ \;'])
        os.chdir(folder + "/ctakes/note_output/")
        print os.getcwd()
        subprocess.check_output(['bash', '-c', 'find . -name "*.xml" -exec mv {} ' + folder + '/xml/ \;'])
        os.chdir(folder)
        subprocess.check_output(['bash', '-c', "rm -rf ctakes/"])    
    print time.time() - t


def RunMetaMap(folder, mmapDir, erisOne, threshold=0, wsd=True, semantic=False, cui=False):
    print "Running MetaMap"
    os.chdir(mmapDir)
    os.system("./skrmedpostctl start; ./wsdserverctl start")
    
    t = time.time()
    os.chdir(folder)
    fileList = [f for f in os.listdir(folder) if f.endswith(".res")]
    res = {}
    for f in range(0, len(fileList)-1):
        print f
        note = open(fileList[f], 'r').read()
        note = note.split('\r\n')
        tmp = []
        for sent in xrange(0, len(note)):
            if note[sent] != '':
                sentence = note[sent].lower().rstrip('\r\n').replace("\r\n", " ")
                sentence = sentence.replace("\n", " ")
                rmPunct = string.maketrans(string.punctuation, ' '*len(string.punctuation))
                sentence = str(sentence).translate(rmPunct)
                cwd = os.getcwd()
                os.chdir(mmapDir)
                if wsd == True:
                    out = commands.getstatusoutput("echo " + sentence + " | ./metamap -N -y --prune 20 -r " + str(threshold))[1]
                else:
                    out = commands.getstatusoutput("echo " + sentence + " | ./metamap -N --prune 20 -r " + str(threshold))[1]
                os.chdir(cwd)
                out = out.split('\n')
                try:
                    tmp.extend([i for i in out if i.startswith('00000000|MMI|')])
                except AttributeError:
                    next
        res[fileList[f]] = tmp
                
    df = pd.DataFrame(res.items(), columns= ['fileName', 'metamap'])
    df.to_csv('metamap_parsed.txt', sep='\t')
    print time.time() - t

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

Title   : Clinical Document Classification Pipeline: Converter module
Author  : Wei-Hung Weng
Created : 10/21/2016
'''

import os, time, string
import xmltodict
import pandas as pd


def Converter(workdir, folder, format):
    if format == "xml":
        fileList = [f for f in os.listdir(folder) if f.endswith(".xml")]
        res = {}
        t = time.time()
        for f in fileList:
            print "cTAKES XML files processing: " + f
            with open(folder + f) as ff:
                doc = xmltodict.parse(ff.read())
                
                try:
                    bow = doc['CAS']['uima.cas.Sofa']['@sofaString']
                    bow = bow.replace('\n', ' ').replace('\r', '').replace('\t', ' ').replace(r'<br />', '').replace(r'<p>', '').replace(r'</p>', '').replace(r'<b>', '').replace(r'</b>', '').replace('START_OF_RECORD=1||||1||||', '').replace('||||END_OF_RECORD', '')
                    bow = bow.lower().encode('utf-8')
                    
                    concept = doc['CAS']['org.apache.ctakes.typesystem.type.refsem.UmlsConcept']
                    l = []
                    for i in xrange(len(concept)):
                        if concept[i]['@codingScheme'] == 'SNOMEDCT':
                            prefix = 'S'
                        elif concept[i]['@codingScheme'] == 'RXNORM':
                            prefix = 'R'
                        l.append([concept[i]['@cui'], concept[i]['@tui'], prefix + concept[i]['@code'], concept[i]['@codingScheme']])
                    
                    umls = []
                    sg = []
                    st = []
                    snomed = []
                    
                    selected_sg = ["T017", "T029", "T023", "T030", "T031", "T022", "T025", "T026", \
                    "T018", "T021", "T024", "T020", "T190", "T049", "T019", "T047", \
                    "T050", "T033", "T037", "T048", "T191", "T046", "T184", "T060", \
                    "T065", "T058", "T059", "T063", "T062", "T061", "T038", "T069", \
                    "T068", "T034", "T070", "T067", "T116", "T195", "T123", "T122", \
                    "T103", "T120", "T104", "T200", "T196", "T126", "T131", "T125", \
                    "T129", "T130", "T197", "T114", "T109", "T121", "T192", "T127"]
                    
                    selected_st = ['T017', 'T022', 'T023', 'T033', 'T034', \
                                'T047', 'T048', 'T049', 'T059', 'T060', \
                                'T061', 'T121', 'T122', 'T123', 'T184']
                    
                    for item in l:
                        umls.append(item[0])
                        if item[1] in selected_sg:
                            sg.append(item[0])
                        if item[1] in selected_st:
                            st.append(item[0])
                        if item[3] == 'SNOMED':
                            snomed.append(item[2])
                    
                    umls = ' '.join(list(set(umls)))
                    sg = ' '.join(list(set(sg)))
                    st = ' '.join(list(set(st)))
                    snomed = ' '.join(list(set(snomed)))
                    
                except KeyError:
                    bow = snomed = umls = sg = st = 'N'
                        
            res[f] = [f, bow, snomed, umls, sg, st]
            
        df = pd.DataFrame.from_dict(res).transpose()
        df.columns = ['fname', 'bow', 'snomed', 'umls', 'sg', 'st']
        df.to_csv(workdir + 'data/data.txt', sep='\t', index=False)
        print "Processing time: " + str(time.time() - t) + " sec"
        
    elif format == "txt" or format == "res":
        if format == "txt":
            fileList = [f for f in os.listdir(folder) if f.endswith(".txt")]
        elif format == "res":
            fileList = [f for f in os.listdir(folder) if f.endswith(".res")]
        res = {}
        t = time.time()
        for f in fileList:
            print "Text files processing: " + f
            bow = open(folder + f, 'r').read().replace('\n', ' ').replace('\r', '').replace('\t', ' ').replace(r'<br />', '').replace(r'<p>', '').replace(r'</p>', '').replace(r'<b>', '').replace(r'</b>', '').replace('START_OF_RECORD=1||||1||||', '').replace('||||END_OF_RECORD', '')
            bow = bow.lower().encode('utf-8')
            res[f] = bow
            
        df = pd.DataFrame.from_dict(res.items())
        df.columns = ['fname', 'bow']
        df.to_csv(workdir + 'data/data.txt', sep='\t', index=False)
        print "Processing time: " + str(time.time() - t) + " sec"
        
    elif format == "cas":
        t = time.time()
        # cas format processing
        print "Processing time: " + str(time.time() - t) + " sec"
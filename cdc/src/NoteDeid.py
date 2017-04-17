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

Title   : Clinical Document Classification Pipeline: Deidentification module (deid)
Author  : Wei-Hung Weng
Created : 10/21/2016
'''

import sys, os, time
import subprocess
import commands
import string
import pandas as pd


def RunDeid(folder, deidDir, rpdr=False, erisOne=False):
    print "Copying deid program into folders"
    cwd = os.getcwd()
    os.system("cp -r " + deidDir + "/* " + folder)
    #subprocess.check_output(['bash', '-c', cmd])

    print "Executing deid"
    if rpdr:
        cmd = "for file in *.txt; do sed 1,2d \"${file}\" > temp && mv temp \"${file}\"; echo '\r\nSTART_OF_RECORD=1||||1||||' | cat - \"$file\" > temp && mv temp \"$file\"; echo '||||END_OF_RECORD\r\n' >> \"$file\"; mv -- \"${file}\" \"${file}.text\"; perl deid.pl \"${file%%.txt.text}\" deid-output.config; sed 's/\[\*\*.*\*\*\]//g' \"${file}.res\" > temp && mv temp \"${file}.res\"; done"
    else:
        cmd = "for file in *.txt; do echo '\r\nSTART_OF_RECORD=1||||1||||' | cat - \"$file\" > temp && mv temp \"$file\"; echo '||||END_OF_RECORD\r\n' >> \"$file\"; mv -- \"${file}\" \"${file}.text\"; perl deid.pl \"${file%%.txt.text}\" deid-output.config; sed 's/\[\*\*.*\*\*\]//g' \"${file}.res\" > temp && mv temp \"${file}.res\"; done"
    os.chdir(folder)
    subprocess.check_output(['bash', '-c', cmd])
    os.system('rm -rf dict; rm -rf doc; rm -rf GSoutput; rm -rf GSstat; rm -rf lists')
    os.system('find . ! -name "*.res" -exec rm -r {} \;')
    os.chdir(cwd)

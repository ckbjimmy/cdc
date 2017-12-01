## Clinical Document Classification Pipeline

- Written by Wei-Hung Weng (HMS, MGH)
- Created: Nov 9, 2016
- Latest update: Nov 27, 2017
- Please contact the author with errors found.
- ckbjimmy {AT} mit {DOT} edu

### Quick Start
1. `git clone` the repository
2. `python setup.py install`
3. Go to the directory
4. Run `sh test_model.sh`

### Word embedding-CRNN
1. Download fasttext embedding from [fasttext website](https://fasttext.cc/docs/en/english-vectors.html) (Use either `wiki-news-300d-1M.vec` or `wiki-news-300d-1M-subword.vec`, depends on your text)
2. use `python EmbCRNN.py [text_path] [label_path] [embedding_path]`

### Paragraph vector
1. use `python Doc2vec.py [text_path] [label_path]`

If you use this code, please kindly cite the paper for this GitHub project (see below for BibTex):

```
@article{weng2017medical,
    title   = {Medical Subdomain Classification of Clinical Notes Using a Machine Learning-Based Natural Language Processing Approach.},
    author  = {Weng, Wei-Hung and Wagholikar, Kavishwar B. and McCray, Alexa T. and Szolovits, Peter and Chueh, Henry C.},
    journal = {BMC Medical Informatics and Decision Making.},
    year    = {2017},
    number  = {17},
    page    = {155},
    note    = {\mbox{doi}:\url{10.1186/s12911-017-0556-8}}
}
```

The code belongs to Wei-Hung Weng and Laboratory of Computer Science, Massachusetts General Hospital.
import pandas as pd

def RPDR(path, lab, num):
    print "Load RPDR data"
    rpdr_dir = "/Users/weng/_hms_phi/rpdr/data/"
    #rpdr_dir = "/PHShome/ww957/cdc_exp_rpdr/data/"
    df = pd.read_csv(rpdr_dir + "rpdr.txt", sep='\t')
    df = df[['fname', 'bow', 'snomed', 'umls', 'sg', 'st']] 
    
    print "Load filename and labels"
    main = pd.read_csv(rpdr_dir + "_rpdr_subset.txt", sep='\t', header=None)
    main.columns = ['fname', 'empi', 'date', 'author', 'before2012', 'noContent', 'authorLast', \
        'authorFirst', 'spId', 'spDesc', 'sex', 'age', 'sp', 'unk', 'refer', 'label']
        
    #for csail server
    #main = pd.read_csv('rpdr_label.txt', sep=',')
    #main.columns = ['X', 'sp', 'spDesc']

    print "RPDR label: " + lab, str(num)
    if lab == "sp":
        y_unencoded = main.spDesc
        print "Subset top " + str(num) + " specialties"
        s = y_unencoded.value_counts()[:int(num)]
        print s
        idx = main[main['spDesc'].isin(s.index) & main['sp'].isin([1])].index
        y_unencoded = y_unencoded.iloc[idx]
        df = df.iloc[idx]
    elif lab == "exp":
        y_unencoded = main.author
        print "Subset the expert with > " + str(num) + " notes"
        s = y_unencoded.value_counts()[y_unencoded.value_counts() > int(num)]
        print s
        idx = main[main['author'].isin(s.index) & main['sp'].isin([1])].index
        y_unencoded = y_unencoded.iloc[idx]
        df = df.iloc[idx]
    
    #s = y_unencoded.value_counts()
    print "Total " + str(len(s)) + " specialties/experts"
    print "Total " + str(len(idx)) + " documents"
    
    df.to_csv(path + 'data/data.txt', sep='\t', index=False)
    y_unencoded.to_csv(path + 'data/label.txt', sep='\t', index=False)
from datasets import load_dataset
import numpy as np
import pandas as pd
import ipdb

'''
parameters:
    - max_labels - the maximum labels assign to a sentences in the dataset

'''
class MyDatasets:
    def __init__(self,dataset_path,test_size=0.2,max_labels=7):
        ds = load_dataset('csv', data_files=dataset_path,encoding = "ISO-8859-1")
        res=ds['train'].train_test_split(test_size=test_size)

        # get the dataset
        train_dataset=res['train']
        test_dataset=res['test']
        self.ds_cols=list(ds['train'].features.keys())

        # some time the dataset accidentally include null rows, so we remove them
        self.train_dataset=train_dataset.filter(lambda elem: any(elem[col] is not None for col in self.ds_cols))
        self.test_dataset=test_dataset.filter(lambda elem: any(elem[col] is not None for col in self.ds_cols))

        self.max_labels=max_labels
    
    def get_classes(self,):
        # collect all classes names from the dataset
        classes=[]
        for i in range(self.max_labels):
            lst1=list(set(self.train_dataset['Topic '+str(i+1)]))
            lst2=list(set(self.test_dataset['Topic '+str(i+1)]))
            lst=lst1+lst2
            classes=classes+lst
            classes=list(set(classes))

        if None in classes:
            classes.remove(None)

        if " " in classes:
            classes.remove(" ")
        
        self.clasess=classes
        return self.clasess

    def get_datasets(self,):
        return self.train_dataset,self.test_dataset
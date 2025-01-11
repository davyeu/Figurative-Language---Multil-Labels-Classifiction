from transformers import AutoTokenizer
import ipdb

class MyTokenization:
    def __init__(self,model_path,classes,class2id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.classes=classes
        self.class2id=class2id
       

    '''
    go over the topics of the datasets and find the labels for each sentences
    then create list of indicator labels for each sentences with lenght of number of labels
    In the next step the tokenizer tokeinzed the text and attach him the relveant labels lists
    '''
    def preprocess_function(self,example):
        text = example['Line']
        keys=example.keys()
        all_labels=[example[key] for key in keys if key.startswith("Topic") and example[key] is not None ]
        labels = [0. for i in range(len(self.classes))]
        for label in all_labels:

            label_id = self.class2id[label]
            labels[label_id] = 1.

        example = self.tokenizer(text, truncation=True)
        example['labels'] = labels
        return example
    
    def df_tokenization(self,ds):
        return ds.map(self.preprocess_function)

        
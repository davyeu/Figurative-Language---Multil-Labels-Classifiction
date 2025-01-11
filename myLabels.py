import ipdb
class MyLabels:
    def __init__(self,max_labels=7,
                 limited_labels=True):
        self.max_labels=max_labels
        self.limited_labels=limited_labels

    '''
        set labels and return updated datasets with updated classes
        if label_list is None then the chosen labels will be those with the x-first labels
        by their ocurances in the dataset, where x defined to be e_idx-s_idx
        else : the chosen lables will be as mention at the list
        that mean that only sentences that have that label will considered in the evaluted dataset.

    '''          
    def set_labels(self,classes,train_dataset,test_dataset,s_idx,e_idx,label_list='None'):
        if(self.limited_labels):
                self.chosen_labels,self.sorted_class_cnt_dict= self.dict_with_selected_labels(classes,
                                                                    train_dataset,test_dataset,s_idx,e_idx,label_list)
                train_dataset=train_dataset.map(self.change_labels)
                test_dataset=test_dataset.map(self.change_labels)
                #remove sentences without labels at all
                train_dataset = train_dataset.filter(lambda elem: any(elem[f"Topic {i+1}"] is not None for i in range(self.max_labels)))
                test_dataset = test_dataset.filter(lambda elem: any(elem[f"Topic {i+1}"] is not None for i in range(self.max_labels)))
        else:
            self.chosen_labels=classes
        #set the dicts for classes
        self.cur_classes=self.chosen_labels
        self.class2id = {class_:id for id, class_ in enumerate(self.cur_classes)}
        self.id2class = {id:class_ for class_, id in self.class2id.items()}

        return train_dataset,test_dataset

    def get_chosen_labels(self,):
        return self.chosen_labels

    def get_num_of_chosen_labels(self,):
        return len(self.chosen_labels)
    
    def change_labels(self,example):
        for i in range(self.max_labels):
            if example['Topic '+str(i+1)] not in self.chosen_labels:
                example['Topic '+str(i+1)]=None
        return example
    
    '''
    return list of x  selected labels where x determinded by e_idx-s_idx
    and sorted dict where is keys is labels and the values are the sentences with this label
    '''
    def dict_with_selected_labels(self,classes,train_dataset,test_dataset,s_idx,e_idx,label_list='None'):
        classes_cnt={class_:0 for class_ in classes}
        for i in range(self.max_labels):
            for label in train_dataset['Topic '+str(i+1)]:
                if label is not None and label !="" and label!=" " :
                    classes_cnt[label]+=1
            for label in test_dataset['Topic '+str(i+1)]:
                if label is not None and label !="" and label!=" " :
                    classes_cnt[label]+=1
        #sort the dict of classes counts and take the top labels
        self.sorted_class_cnt_dict= {k: v for k, v in sorted(classes_cnt.items(), key=lambda item: item[1],reverse=True)}
        if(self.limited_labels):
            if label_list is 'None':
                self.chosen_labels=list(self.sorted_class_cnt_dict.keys())[s_idx:e_idx]
            else:
                self.chosen_labels=label_list
        else:
            self.chosen_labels=list(self.sorted_class_cnt_dict.keys())[0:len(classes)]
        return self.chosen_labels,self.sorted_class_cnt_dict
    
    '''
    return dict with sents(value) by labels (keys)
    '''
    def get_dict_sents(self,train_dataset,test_dataset):
        def set_dict_labels_by_ds(self,ds,dict_labels):
            # append sents for each lable related list
            for i in range(len(ds['Line'])):
                dict_sent=ds[i]
                sent=dict_sent['Line']
                sent_labels=[dict_sent[key] for key in dict_sent.keys() if key.startswith('Topic') and dict_sent[key] is not None]
                for lab in sent_labels:
                    dict_labels[lab].append(sent)
            return dict_labels
        
        # create a dict where is keys is the labels and the value for each key is empty set
        labels_=self.chosen_labels
        dict_labels={lab: [] for lab in labels_}
        dict_labels=set_dict_labels_by_ds(self,train_dataset,dict_labels)
        dict_labels=set_dict_labels_by_ds(self,test_dataset,dict_labels)

        #sorted the dict by the number of sentences per label
        s_keys=sorted(dict_labels.keys(),key=lambda label:len(dict_labels[label]),reverse=True)
        self.sorted_dict_sents={k:dict_labels[k] for k in s_keys}

        return self.sorted_dict_sents   


    def get_sents_by_label (self,label,max_num=None):
        try:
            label_num=len(self.sorted_dict_sents[label])
            if max_num is None or max_num>label_num or max_num<0:
                max_num=label_num            
            return self.sorted_dict_sents[label][0:max_num]
        except:
            print(f"sorted_dict_sents do not initialized or {label} do not exists" )
    
    '''
    return list of sentences taken from self.sorted_dict_sents  
    where from each label we take "max_val" sentences limited to 
    the number of labels given in num_of_sel_labels
    '''
    def get_selected_sents(self,max_val,num_of_sel_labels):
        dict_sents=self.sorted_dict_sents
        limited_sorted_dict_sent={label: self.get_sents_by_label(label,max_val) for label in dict_sents.keys()}
        lst_of_selected_labels=list(dict_sents.keys())[0:num_of_sel_labels]
        limited_dict_lab_sent_of_selected_labels={label:limited_sorted_dict_sent[label] for label in lst_of_selected_labels}

        sents_of_selected_labels=[ limited_dict_lab_sent_of_selected_labels[label]  for label in lst_of_selected_labels]
        selected_labels_sents=[sent for sents in sents_of_selected_labels for sent in sents ]
        selected_labels_sents=list(set(selected_labels_sents))

        return selected_labels_sents


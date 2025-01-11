from sklearn.metrics import classification_report
import myTrainer,myLabels,myTokenization
import numpy as np
import pandas as pd
import datetime
import ipdb

'''
parameters:
    classes    - list with names of all the classes in the dataset
    model_path - the kind of the model, for example: distilbert-base-uncased
    s_idx      - the start index of the selected lables
    e_idx      - the end index of the selected labels, by defualt set to be 
                 number of choosen labels given by the user
    org_train_dataset - the original train dataset with all the labels in classess
    

'''
class MyEvaluation:
    def __init__(self,classes,model_path,max_labels,limited_labels,
                 org_train_dataset,org_test_dataset,kind_of_task = "multi_label_classification",all_classes='None'):
        self.kind_of_task=kind_of_task  
        self.model_path=model_path
        self.classes=classes
        self.max_labels=max_labels
        self.limited_labels=limited_labels
        self.org_train_dataset=org_train_dataset
        self.org_test_dataset=org_test_dataset

        self.all_classes=all_classes


    
    '''
    this function train only on one datasest

    parameters:
        num_of_chosen_labels - number of labels in the trained dataset
        train_dataset - the dataset that contains only the sentences with selected labels
    '''
    def train_on_single_ds(self,save_strategy="no",load_best_model_at_end=False,output_dir="model_results",
                           s_idx=0,e_idx=10,label_list='None'):

    
        self.mylbs=myLabels.MyLabels(max_labels=self.max_labels,limited_labels=self.limited_labels)
        self.train_dataset,self.test_dataset=self.mylbs.set_labels(self.classes,self.org_train_dataset,
                self.org_test_dataset,s_idx,e_idx,label_list)
        
        #set the size of the vector of prediction
        if self.all_classes=='None':
            #in this case only the labels that appeared in the dataset will set the prediction vector
            self.all_classes=self.mylbs.cur_classes
            self.class2id = self.mylbs.class2id
            self.id2class = {id:class_ for class_, id in self.class2id.items()}
            self.cls_names=self.mylbs.get_chosen_labels()
        # in the else statement we set more labels in the prediction vector
        # the perpose of this method is to build itertive modle in size n , that will learn each time
        #  m labels where m<n . so in conclusion after "i" iteration it will know the n labels. (where n=i*m)
        else:
            self.class2id = {class_:id for id, class_ in enumerate(self.all_classes)}
            self.id2class = {id:class_ for class_, id in self.class2id.items()}
            self.cls_names=self.all_classes

        self.mytok=myTokenization.MyTokenization(self.model_path,self.all_classes,self.class2id)
        self.tokenized_train_dataset =self.mytok.df_tokenization(self.train_dataset)
        self.tokenized_test_dataset = self.mytok.df_tokenization(self.test_dataset)

        self.mytrn=myTrainer.MyTrainer(self.mytok.tokenizer)
        self.model=self.mytrn.model_initialization(self.model_path,
            self.all_classes,self.id2class,
            self.class2id,self.kind_of_task)
        
       
        
        self.training_args=self.mytrn.set_model_training_args(output_dir=output_dir,save_strategy=save_strategy,
                                                              load_best_model_at_end=load_best_model_at_end)
                                                             
        self.trainer=self.mytrn.set_The_Trainer(self.model,
            self.training_args,self.tokenized_train_dataset,self.tokenized_test_dataset,
            self.mytok.tokenizer)
        
        self.test_sents=self.test_dataset['Line']
        
        self.trainer.train()

    def sigmoid(self,x):
            return 1/(1 + np.exp(-x))
    
    '''
        #get the logit_preds and use them for classifcation report
        parameters:
        preds - list of predictions for the test sentence
                every pred in preds is vector of zero and ones (indcator zeros)
                that predicit the labels of each sentences
    '''
    def evaluate_on_single_ds(self):
        self.predictions=self.trainer.predict(self.tokenized_test_dataset)
        logit_preds=self.predictions.predictions
        self.preds=[]
        for lst in logit_preds:
            sent_preds=[]
            for pred in lst:
                activation_func_val=self.sigmoid(pred)
                sent_preds.append((activation_func_val>=0.5).astype(int))
            self.preds.append(sent_preds)
        self.preds=np.array(self.preds)

        #use the preditions of the trainer
        y_pred = self.preds
        y_true = self.predictions.label_ids
        if self.all_classes=='None':
            target_names = self.mylbs.get_chosen_labels()
        else:
            target_names=self.all_classes
        self.classification_report=classification_report(y_true, y_pred, target_names=target_names)

    
    def print_classification_report(self):
        print(self.classification_report)
    
    def get_predictions(self):
        preds_labels_names=[]
        for pred in self.preds:
            preds_labels_names.append([self.cls_names[i] for i in range(len(pred)) if pred[i]==1])

        data_dic={"Content":self.test_sents,"labels":preds_labels_names}
        return data_dic

    def save_predictions(self):         
        preds_labels_names=[]
        for pred in self.preds:
            preds_labels_names.append([self.cls_names[i] for i in range(len(pred)) if pred[i]==1])

        data_dic={"Content":self.test_sents,"labels":preds_labels_names}

        df=pd.DataFrame(data_dic)
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        file_name=f"predictions_{now}.csv"

        df.to_csv("results/"+file_name, index=False)
        print(f"predictions printed to csv file:  {file_name}")

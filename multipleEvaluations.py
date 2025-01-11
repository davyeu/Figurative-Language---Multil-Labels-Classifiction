from sklearn.metrics import classification_report
import myEvaluation
import numpy as np
import pandas as pd
import datetime
import ipdb
import os


class MultipleEvaluations:
    '''
    parameters:
        myeval- object of MyEvaluation class. It represent the general model
                that trained and evaluated on dataset with x labels (for example: x=50)
        classes- list of all classes names in the dataset
        max_labels= the number of maximum labels for each sentences
        limited_labels= boolean variable 
    ''' 
    def __init__(self,myeval,classes,max_labels=7,limited_labels=True,s_idx=0,e_idx='None',
                 saving_model=False,output_dir='./'):

        self.myeval=myeval
        if e_idx=='None':
            e_idx=len(classes)
        
        if (saving_model):
            myeval.train_on_single_ds(save_strategy="epoch",load_best_model_at_end=True,
                                      output_dir=output_dir,s_idx=s_idx,e_idx=e_idx)
        else:
            myeval.train_on_single_ds(s_idx=s_idx,e_idx=e_idx)

        myeval.evaluate_on_single_ds()

        # extract the relevant information from the object of MyEvaluations
        # like the test dataset sentences and its labels.
   
        self.predictions_label_ids=myeval.predictions.label_ids
        self.cur_test_ds=myeval.test_dataset
        self.cur_test_ds_sents=self.cur_test_ds['Line']
        self.cur_train_ds=myeval.train_dataset
        self.cls_names=myeval.cls_names
        self.class2id=myeval.mylbs.class2id
        self.id2class=myeval.mylbs.id2class
        self.classes=classes
        self.max_labels=max_labels
        self.limited_labels=limited_labels

    '''
    parameters
        models_dir= The directory where the trained models are stored
    '''
    def set_evaluated_models_path(self,models_dir,project_path):
        # go to model path
        os.chdir(models_dir)
        cur_dir=os.getcwd()
        models_files_names=os.listdir()
        self.models_files_names=[models_dir+"/"+model_file for model_file in models_files_names ]

        #return to project path
        os.chdir(project_path)
        os.getcwd()

    '''
    evaluate the  trained models stored in model_dir.
    for example: in the case where 5 models stored in the directory,
    each one of them train on 10 different labels, then the "step" variable will be "10".
    So each model train only on sentences that belong to 10 labels corrsponded of the current model.
    evantually we can use the prediction of those 5 models for evalutating the test dataset with
    50 different labels
    '''
    def evaluateModels(self,step=10):
        s_idx,e_idx=0,step
        models_data_dict_lst=[]
        models=[]
        for i in range(len(self.models_files_names)):
            model_path=self.models_files_names[i]
            cur_eval=myEvaluation.MyEvaluation(self.classes,model_path,self.max_labels,self.limited_labels,
                            self.cur_train_ds,self.cur_test_ds)
            cur_eval.train_on_single_ds(s_idx=s_idx,e_idx=e_idx)
            cur_eval.evaluate_on_single_ds()
            models.append(cur_eval)
            models_data_dict_lst.append(cur_eval.get_predictions())
            s_idx,e_idx=s_idx+step,e_idx+step

        #convert to df and concatnate the data_ditc for each model
        models_df_lst=[pd.DataFrame(data_dict)for data_dict in models_data_dict_lst]
        models_df_preds=pd.concat(models_df_lst)
        #sort the the data_dicts by the sentences
        self.models_df_preds=models_df_preds.sort_values(by="Content")

        self.collect_sents_and_labels_from_evaluated_models()

    '''
        collect the sentences and the predicted labels from the 
        datasets that the evaluated models trained on them
        and keep them in two list: sent and labels.
    '''
    def collect_sents_and_labels_from_evaluated_models(self,):
        self.sents,self.labels=[],[]
        for i in range(len(self.models_df_preds)):
            if self.models_df_preds.iloc[i,0] not in self.sents:
                self.sents.append(self.models_df_preds.iloc[i,0])
                sent_labels=self.models_df_preds.iloc[i,1]
                # while the sentences is the same -> append the labels to one list
                while i<len(self.models_df_preds)-1 and self.models_df_preds.iloc[i,0]==self.models_df_preds.iloc[i+1,0]:
                    cur_lst=self.models_df_preds.iloc[i+1,1]
                    if(len(cur_lst)>0):
                        sent_labels=sent_labels+cur_lst
                    i=i+1
                self.labels.append(sent_labels)

        self.set_model_predictions()
    
    def set_model_predictions(self,):
        # now after we collected the labels of all test_ds sentences
        # for each revlant specfic model - we can produce the labels vectors 
        # for the general model. For example if each sepecific model have vector 
        # of labels in length 10, so in the case of 5 models - the general model
        # will have label vector in length of 50.
        model_preds=[]
        for labels_lst in self.labels:
            arr=np.zeros_like(self.myeval.preds[0],dtype='float')
            for lab in labels_lst:
                idx=self.class2id[lab]
                arr[idx]=1
            model_preds.append(arr)
        self.model_preds= np.array(model_preds)

        #now we match the indexes of cur_test_ds that will be compatiable
        cur_test_ds_sents_idx=[self.sents.index(sent) for sent in self.cur_test_ds_sents]
        #now we change the orders of the preds by the indexes
        self.comp_preds=self.model_preds[cur_test_ds_sents_idx]
        
        #convert the 1-indector predection to list of labels
        self.sent_labels_preds=[]
        for pred in self.comp_preds:
            labs=[]
            pred=pred.tolist()
            for i in range(len(pred)):
                if pred[i] ==1:
                    labs.append(self.id2class[i])
            self.sent_labels_preds.append(labs)
    
    def print_classifiction_report(self,):
        print(classification_report(self.predictions_label_ids,
                                    self.comp_preds, target_names=self.cls_names))
   
    def save_predictions_in_csv_file(self,):
        data_dic={"Content":self.cur_test_ds_sents,"labels":self.sent_labels_preds}
        df=pd.DataFrame(data_dic)
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        file_name=f"predictions_{now}.csv"

        df.to_csv("results/"+file_name, index=False)
        print(f"predictions printed to csv file:  {file_name}")
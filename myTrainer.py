from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
import ipdb

class MyTrainer:
    def __init__(self,tokenizer):
        #dynamically pad the sentences to the longest length in a batch
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



    # set the metrics for the model estimations
    def model_metrics(self,sort_of_metric="all"):
        if sort_of_metric=="all":
            self.clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
        else:
            self.clf_metrics = evaluate.load("f1",average="macro")
        

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    '''
    eval_pred:= logit prediction of one iteration of the model
    clf_metrics:= metric object set in  model_metrics function

    '''
    def compute_metrics(self,eval_pred):
        global clf_metrics
        predictions, labels = eval_pred
        predictions = self.sigmoid(predictions)
        predictions = (predictions > 0.5).astype(int).reshape(-1)
        return clf_metrics.compute(predictions=predictions,
                                    references=labels.astype(int).reshape(-1))

    '''
    model_path:= the kind of the model, for example: distilbert-base-uncased
    classes:= list of classes names
    id2class:= dict of ids and classes
    class2id:= dict of classes names and ids
    '''
    def model_initialization(self,model_path,classes,id2class,class2id,
                            kind_of_task = "multi_label_classification"):
        model = AutoModelForSequenceClassification.from_pretrained(model_path,
                num_labels=len(classes),
            id2label=id2class, label2id=class2id,
                problem_type =  kind_of_task
            )
        return model


    # if you does not want to save:
            # save_strategy="no",
            # load_best_model_at_end=False
    # if you want to save the model, change the arguments to be:
            # save_strategy="epoch",
            # load_best_model_at_end=True,
    def set_model_training_args(self,output_dir="model_results",
            overwrite_output_dir = True,
            learning_rate=2e-5,
            per_device_train_batch_size=3,
            per_device_eval_batch_size=3,
            num_train_epochs=4,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False):
        
       
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end)
                                    
        return training_args
    '''
    model:= rv of model_initialization func
    args:=rv of set_model_training_args func
    train_dataset:=tokenized_train_dataset 
    eval_dataset:=tokenized_test_dataset
    data_collator dynamically pad the sentences to the longest length in a batch
    '''
    def set_The_Trainer(self,model,training_args,
        tokenized_train_dataset,tokenized_test_dataset,
        tokenizer):
        trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        data_collator=self.data_collator,
        #compute_metrics=self.compute_metrics
        )
        return trainer

    def train_model(trainer):
        trainer.train()

    def evaluate_model(trainer):
        trainer.evaluate()
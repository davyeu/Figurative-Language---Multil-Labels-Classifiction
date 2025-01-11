# Figurative-Language---Multil-Labels-Classifiction
multi label classifiction of Figurative Language of 17th Century English Plays

The project classifiy Figurative sentences form 17th Century English Plays to one to seven label for each.
beacuse the the number of labels in the relevant datasets is more than 100 labels the initial metric model results is very poor.
By taking unique apporace of ensmble learning we succeded to acheive  highly F1-score,Avg. precsion and recall result as describe in the Table
below.

<h2> Datasets</h2>
The Datasets for the model stored in Datasets folder. It include the raw data taken directly form the relvent 17th Century English Plays and
texts with labels. Two main dataset created From those texts : the small one that include 1389 comment with one to seven labels and additionl information 
for each sentence, and a large Dataset that include 11,523 sentences. The creation of those datasets and their labeling had by reseracher that specifise in 
old English Literture. I do a standartzation for labels of those two datasets and save them as normalized datasets.

<h2>project files</h2>
This project include the main program as Jupyter notbook, configuration file saved as csv file and python file with the clases of the model.
You can run this model from the main Jupyter notbook. Before runing this notbook you can set the relvent configurations in config.csv file.
Explation of each configuration can be found in this file. In this project we use wandb libary for training the model, so you have to set your
personal wandb API key in the config file before running the model. Wandb API key can be generated here 

![image](https://github.com/user-attachments/assets/15ac9b46-c976-4491-bc85-e8860adb6577)# Figurative-Language---Multil-Labels-Classifiction
multi label classifiction of Figurative Language of 17th Century English Plays

The project classifiy Figurative sentences form 17th Century English Plays to one to seven label for each.
beacuse the the number of labels in the relevant datasets is more than 100 labels the initial metric model results is very poor.
By taking unique apporace of ensemble learning we succeded to acheive  highly F1-score,Avg. precsion and recall result as describe in the Table
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
personal wandb API key in the config file before running the model. Wandb API key can be generated form [Visit OpenAI's Website](https://www.openai.com)

[]([url](https://wandb.ai/site))



To use the ensemble model you can download it form  [here. ](https://drive.google.com/drive/folders/1UYlFUJ4LykeEgQfc2K0eEw32Dc8kxdFv?usp=sharing) 
For this method you need to set the to kind of evaluation parametes in config.csv to "multiple evaluations".

<h2>metrics</h2>
Below are the metrics achived by applying the ensemble model on split form the small dataset that include only 50 different labels.

<table border="1" style="border-collapse: collapse; width: 100%;">
    <thead>
        <tr>
            <th>Category</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Support</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>anatomy</td><td>0.99</td><td>0.99</td><td>0.99</td><td>71</td></tr>
        <tr><td>violence</td><td>0.95</td><td>0.78</td><td>0.86</td><td>51</td></tr>
        <tr><td>politics</td><td>0.98</td><td>0.95</td><td>0.97</td><td>44</td></tr>
        <tr><td>emotions</td><td>1.00</td><td>0.96</td><td>0.98</td><td>52</td></tr>
        <tr><td>religion</td><td>0.97</td><td>0.93</td><td>0.95</td><td>42</td></tr>
        <tr><td>mental_faculty__state__entities</td><td>0.95</td><td>0.98</td><td>0.96</td><td>41</td></tr>
        <tr><td>nature</td><td>0.94</td><td>0.89</td><td>0.92</td><td>38</td></tr>
        <tr><td>spatial</td><td>0.89</td><td>0.78</td><td>0.83</td><td>41</td></tr>
        <tr><td>economics</td><td>1.00</td><td>0.96</td><td>0.98</td><td>24</td></tr>
        <tr><td>character_traits__nature</td><td>0.92</td><td>0.71</td><td>0.80</td><td>34</td></tr>
        <tr><td>food</td><td>0.95</td><td>0.78</td><td>0.86</td><td>27</td></tr>
        <tr><td>language</td><td>0.93</td><td>0.96</td><td>0.95</td><td>28</td></tr>
        <tr><td>time</td><td>0.93</td><td>0.96</td><td>0.94</td><td>26</td></tr>
        <tr><td>sexuality</td><td>0.95</td><td>0.80</td><td>0.87</td><td>25</td></tr>
        <tr><td>animals</td><td>1.00</td><td>0.94</td><td>0.97</td><td>16</td></tr>
        <tr><td>death</td><td>0.93</td><td>0.78</td><td>0.85</td><td>18</td></tr>
        <tr><td>social_relations</td><td>0.86</td><td>0.90</td><td>0.88</td><td>20</td></tr>
        <tr><td>social_status</td><td>1.00</td><td>0.72</td><td>0.84</td><td>18</td></tr>
        <tr><td>ethics</td><td>0.76</td><td>0.87</td><td>0.81</td><td>15</td></tr>
        <tr><td>recognition</td><td>1.00</td><td>0.83</td><td>0.91</td><td>6</td></tr>
        <tr><td>medicine</td><td>1.00</td><td>1.00</td><td>1.00</td><td>7</td></tr>
        <tr><td>judiciary</td><td>0.95</td><td>1.00</td><td>0.97</td><td>18</td></tr>
        <tr><td>architecture</td><td>0.91</td><td>0.91</td><td>0.91</td><td>11</td></tr>
        <tr><td>mystical</td><td>1.00</td><td>0.91</td><td>0.95</td><td>11</td></tr>
        <tr><td>privation</td><td>0.93</td><td>0.82</td><td>0.88</td><td>17</td></tr>
        <tr><td>movement</td><td>0.88</td><td>1.00</td><td>0.93</td><td>7</td></tr>
        <tr><td>familial</td><td>1.00</td><td>1.00</td><td>1.00</td><td>17</td></tr>
        <tr><td>quantities</td><td>1.00</td><td>1.00</td><td>1.00</td><td>6</td></tr>
        <tr><td>sensations</td><td>1.00</td><td>1.00</td><td>1.00</td><td>9</td></tr>
        <tr><td>clothes</td><td>1.00</td><td>0.57</td><td>0.73</td><td>7</td></tr>
        <tr><td>life</td><td>1.00</td><td>1.00</td><td>1.00</td><td>12</td></tr>
        <tr><td>women</td><td>1.00</td><td>1.00</td><td>1.00</td><td>7</td></tr>
        <tr><td>sleep</td><td>1.00</td><td>1.00</td><td>1.00</td><td>8</td></tr>
        <tr><td>military</td><td>1.00</td><td>0.50</td><td>0.67</td><td>4</td></tr>
        <tr><td>suffering</td><td>0.92</td><td>0.92</td><td>0.92</td><td>12</td></tr>
        <tr><td>feelings</td><td>0.77</td><td>1.00</td><td>0.87</td><td>10</td></tr>
        <tr><td>sounds</td><td>1.00</td><td>0.88</td><td>0.93</td><td>8</td></tr>
        <tr><td>darkness</td><td>1.00</td><td>1.00</td><td>1.00</td><td>9</td></tr>
        <tr><td>weapons__armor</td><td>1.00</td><td>1.00</td><td>1.00</td><td>6</td></tr>
        <tr><td>physical_activities</td><td>1.00</td><td>0.50</td><td>0.67</td><td>10</td></tr>
        <tr><td>danger__safety</td><td>1.00</td><td>0.86</td><td>0.92</td><td>7</td></tr>
        <tr><td>consumption</td><td>0.50</td><td>0.22</td><td>0.31</td><td>9</td></tr>
        <tr><td>etiquette</td><td>0.80</td><td>0.67</td><td>0.73</td><td>6</td></tr>
        <tr><td>cleaning</td><td>0.88</td><td>0.88</td><td>0.88</td><td>8</td></tr>
        <tr><td>appearance</td><td>0.80</td><td>0.89</td><td>0.84</td><td>9</td></tr>
        <tr><td>fire</td><td>1.00</td><td>0.89</td><td>0.94</td><td>9</td></tr>
        <tr><td>reproduction</td><td>1.00</td><td>1.00</td><td>1.00</td><td>8</td></tr>
        <tr><td>colors</td><td>1.00</td><td>0.86</td><td>0.92</td><td>7</td></tr>
        <tr><td>destruction</td><td>0.67</td><td>0.50</td><td>0.57</td><td>4</td></tr>
        <tr><td>games__sports</td><td>1.00</td><td>0.50</td><td>0.67</td><td>4</td></tr>
        <tr><td><b>micro avg</b></td><td><b>0.95</b></td><td><b>0.88</b></td><td><b>0.91</b></td><td><b>904</b></td></tr>
        <tr><td><b>macro avg</b></td><td><b>0.94</b></td><td><b>0.85</b></td><td><b>0.89</b></td><td><b>904</b></td></tr>
        <tr><td><b>weighted avg</b></td><td><b>0.95</b></td><td><b>0.88</b></td><td><b>0.91</b></td><td><b>904</b></td></tr>
        <tr><td><b>samples avg</b></td><td><b>0.95</b></td><td><b>0.89</b></td><td><b>0.92</b></td><td><b>904</b></td></tr>
    </tbody>
</table>



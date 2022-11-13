#############################################################################################
#### Information Retrieval (CS60092, Autumn 2022-23) Term Project (Part 2,3)###################           
#### Project Code: DEFAULT IR-TP#############################################################
#### Group No: 9 ############################################################################
#############################################################################################


Dataset used:
Part2 and 3: https://drive.google.com/drive/folders/1J5M8IWk-ERC4d7jR-5dJ3WjmkoKfLE8j

Vocabulary: 45k terms

Requirements :

ipykernel==6.15.2
ipython==8.5.0
ipython-genutils==0.2.0
jupyter-client==7.3.5
jupyter-core==4.11.1
jupyterlab-pygments==0.2.2
matplotlib==3.5.3
matplotlib-inline==0.1.6
mypy-extensions==0.4.3
nltk==3.7
notebook==6.4.12
numpy==1.23.3
pandas==1.4.4
python-apt==2.0.0+ubuntu0.20.4.8
python-dateutil==2.8.2
python-debian===0.1.36ubuntu1


Python Settings:
The interpreter we have used is Python3.8.10


Running the Code:

2A:

Run:
python3 Assignment2_9_ranker.py ./Data/CORD-19 model_queries_9.bin

Link to the inverted index: https://drive.google.com/file/d/1FdkKyaPnSIgAKvUZ-6n3QgHky6LBWdxz/view?usp=sharing

Files generated:
Assignment2_9_ranked_list_A.csv
Assignment2_9_ranked_list_B.csv
Assignment2_9_ranked_list_C.csv


2B: 

Run:
python Assingment2_9_evaluator.py ./Data/qrels.csv Assignment2_9_ranked_list_A.csv
python Assingment2_9_evaluator.py ./Data/qrels.csv Assignment2_9_ranked_list_B.csv
python Assingment2_9_evaluator.py ./Data/qrels.csv Assignment2_9_ranked_list_C.csv

Files generated:

Assignment2_9_metrics_A.txt
Assignment2_9_metrics_B.txt
Assignment2_9_metrics_C.txt

Remarks:

Depending on the ranked list file passed in the command, the respective metric file is generated


3A:
Run:
python3 Assignment3_9_rocchio.py ./Data/CORD-19 model_queries_9.bin ./Data/qrels.csv Assignment2_9_ranked_list_A.csv 

Files generated:
Assignment3_9_rocchio_RF_metrics.csv
Assignment3_9_rocchio_PsRF_metrics.csv

Added Files:
Assignment3_9_rocchio_report.txt

Running time:
The code took almost 2 hours to run.


3B:
Run:
python3 Assignment3_9_important_words.py ./Data/CORD-19 model_queries_9.bin 

Files generated:
Assignment2_9_ranked_list_A.csv







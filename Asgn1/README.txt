#############################################################################################
#### Information Retrieval (CS60092, Autumn 2022-23) Term Project (Part 1)###################           
#### Project Code: DEFAULT IR-TP#############################################################
#### Group No: 9 ############################################################################
#############################################################################################

Dataset:
Link: https://drive.google.com/drive/folders/1J5M8IWk-ERC4d7jR-5dJ3WjmkoKfLE8j
Download the three files and put them in a sub-folder called "Data" 

Requirements (also mentioned in the file 'requirements.txt'):

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
pickleshare==0.7.5
python-apt==2.0.0+ubuntu0.20.4.8
python-dateutil==2.8.2
python-debian===0.1.36ubuntu1


Python Settings:
The interpreter we have used is Python3.8.10


Running the Code:
1A
python3 Assignment1_9_indexer_final.py <path to the CORD-19 folder>
python3 Assignment1_9_indexer_final.py ./Data/CORD-19

Files generated:
    model_queries_9.bin

1B
python3 Assignment1_9_parser.py <path to the query file>
python3 Assignment1_9_parser.py ./Data/queries.csv

Files generated:
    queries_9.txt

1C
python3 Assignment1_9_bool.py <path to model> <path to query file>
python3 Assignment1_9_bool.py ./model_queries_9.bin ./queries_9.txt

Files generated:
    results.txt


Remarks:
Part 1A takes around 20-25 mins to run and parse the 50,000+ files.
The Data folder (downloadable folder and of very large size) needs to be placed in the repository




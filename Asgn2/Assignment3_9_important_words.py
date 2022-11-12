import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import numpy as np
import math


nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")
nltk.download("punkt")
lemmatizer = WordNetLemmatizer()
punc_tokenizer = nltk.RegexpTokenizer(r"\w+")
stop_words = set(stopwords.words("english"))

import os, json
import pandas as pd
import pickle
import sys

path_to_json = sys.argv[1]
path_to_index = sys.argv[2]
path_to_rankeddocs = sys.argv[3]


if not os.path.exists(path_to_json):
    print("Error : Data Folder doesn't Exist ")
    sys.exit()

if not os.path.exists(path_to_index):
    print("Error : Model doesn't Exist ")
    sys.exit()

if not os.path.exists(path_to_rankeddocs):
    print("Error : Ranked List doesn't Exist ")
    sys.exit()


json_files = [pos_json for pos_json in os.listdir(path_to_json)]

raw_dict = dict()

for filename in json_files:
    with open(path_to_json + "/" + filename, "r") as f:
        data = json.load(f)
        for row in data["abstract"]:
            if data["paper_id"] in raw_dict:
                raw_dict[data["paper_id"]] += row["text"] + " "
            else:
                raw_dict[data["paper_id"]] = row["text"] + " "
        if data["paper_id"] in raw_dict:
            raw_dict[data["paper_id"]] = raw_dict[data["paper_id"]][:-1]
        else:
            raw_dict[data["paper_id"]] = ""

cord_to_paperId = dict()
paper_to_cordId = dict()
corpus = dict()

print("Reading Done!")

idmap = pd.read_csv("./Data/id_mapping.csv")

for index, row in idmap.iterrows():
    if(row["paper_id"] in raw_dict):
        cord_to_paperId[row["cord_id"]] = row["paper_id"]
        paper_to_cordId[row["paper_id"]] = row["cord_id"]
        if row["cord_id"] not in corpus:
            corpus[row["cord_id"]] = raw_dict[row["paper_id"]]
        else:
            corpus[row["cord_id"]] = corpus[row["cord_id"]]+" "+raw_dict[row["paper_id"]]

# preprocessing
processed_corpus = dict()
for doc in corpus:
    text = corpus[doc]
    word_tokens = word_tokenize(text)
    word_tokens = [w.lower() for w in word_tokens]
    removed_stopwords = [
        w for w in word_tokens if not w.lower() in stop_words
    ]  # Removing all the stopwords in the queries
    sent_removed = " ".join(removed_stopwords)
    only_words = punc_tokenizer.tokenize(sent_removed)
    lemmatized_output = " ".join([lemmatizer.lemmatize(w) for w in only_words])
    processed_corpus[doc] = lemmatized_output

print("Preprocessing Done!")



f = open(path_to_index, "rb")
InvertedIndex = pickle.load(f)

DocumentFrequency = dict()

for keys in InvertedIndex:
  if(len(InvertedIndex[keys])>2):
    DocumentFrequency[keys] = len(InvertedIndex[keys])
  

print("DF Done!")

TermFrequency = defaultdict(dict)  #finding TF part  #defaultdict allows for 2d (or 3d or wtv)keys 

for doc in processed_corpus:
    for word in processed_corpus[doc].split(" "):
        if(word in DocumentFrequency):
            if((doc in TermFrequency)&(word in TermFrequency[doc])):
                TermFrequency[doc][word]+=1
            else:
                TermFrequency[doc][word]=1

print("TF Done!")

# Mapping Each Term to a integer between 0 to |V|
keynum = dict()
numtokey = dict()
count = 0
for keys in DocumentFrequency:
    keynum[keys] = count
    numtokey[count] = keys
    count = count + 1


print("Mapping Done!")


rankeddocs = pd.read_csv(path_to_rankeddocs)
querydocs = rankeddocs['document id']
Top10Docs = dict()
for i in range(len(querydocs)):
    docs = querydocs[i].split(' ')
    docs = docs[:10]
    Top10Docs[i] = []
    for doc in docs:
        Top10Docs[i].append(doc)

# TF-IDF of documents (lnc)
lncDoc = dict()
for queryid in Top10Docs:
    for doc in Top10Docs[queryid]:
        if doc not in lncDoc:
            lncDoc[doc] = np.zeros(len(keynum))
            for key in TermFrequency[doc]:
                lncDoc[doc][keynum[key]] = TermFrequency[doc][key]
                if(lncDoc[doc][keynum[key]] > 0):
                    lncDoc[doc][keynum[key]] = 1 + math.log10(lncDoc[doc][keynum[key]])
            lncDoc[doc] /= np.sqrt(np.sum(lncDoc[doc]**2))

print("LNC Done!")

avglnc = dict()
for queryid in Top10Docs:
    avglnc[queryid] = np.zeros(len(keynum))
    for doc in Top10Docs[queryid]:
        avglnc[queryid] += lncDoc[doc]
    avglnc[queryid]/=len(Top10Docs[queryid])



def WriteDictToFile(my_dict,csv_file):
    try:
        with open(csv_file, 'w') as f:
            f.write('{0},{1}\n'.format('query id', 'important words'))
            [f.write('{0},{1}\n'.format(key, value)) for key, value in (my_dict.items())]
    except IOError:
        print("I/O error")


word_ranks = dict()
for queryid in avglnc:
    word_list=list()
    for idx, score in np.ndenumerate(avglnc[queryid]):
        word_list.append([score,numtokey[idx[0]]])
    word_list.sort(reverse=True)
    word_ranks[queryid]=' '.join([x[1] for x in word_list[:5]])

WriteDictToFile(word_ranks,"Assignment3_9_important_words.csv")

print("Writing File Done!")
        







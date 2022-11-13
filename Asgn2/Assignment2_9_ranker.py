import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import numpy as np
from numpy import dot
from numpy.linalg import norm
import math
import gc
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


if not os.path.exists(path_to_json):
    print("Error : Data Folder doesn't Exist ")
    sys.exit()

if not os.path.exists(path_to_index):
    print("Error : Model doesn't Exist ")
    sys.exit()



f = open(path_to_index, "rb")
InvertedIndex = pickle.load(f)

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

del(corpus)
del(raw_dict)
del(cord_to_paperId)
del(paper_to_cordId)
gc.collect()

DocumentFrequency = dict()

for keys in InvertedIndex:
  if(len(InvertedIndex[keys])>2):
    DocumentFrequency[keys] = len(InvertedIndex[keys])
  

print("DF Done!")

TermFrequency = defaultdict(dict)  #finding TF part  

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
count = 0
for keys in DocumentFrequency:
    keynum[keys] = count
    count = count + 1


print("Mapping Done!")

# TF-IDF of documents (lnc)
lncDoc = dict()
for doc in TermFrequency:
    lncDoc[doc] = np.zeros(len(keynum))
    for key in TermFrequency[doc]:
        lncDoc[doc][keynum[key]] = TermFrequency[doc][key]
        if(lncDoc[doc][keynum[key]] > 0):
            lncDoc[doc][keynum[key]] = 1 + math.log10(lncDoc[doc][keynum[key]])
    lncDoc[doc] /= np.sqrt(np.sum(lncDoc[doc]**2))

print("LNC Done!")

# TF-IDF of documents (anc)
ancDoc = dict()
for doc in TermFrequency:
    ancDoc[doc] = np.zeros(len(keynum))
    for key in TermFrequency[doc]:
        ancDoc[doc][keynum[key]] = TermFrequency[doc][key]
    ancDoc[doc] = 0.5 + 0.5*ancDoc[doc]/np.max(ancDoc[doc])
    ancDoc[doc] /= np.sqrt(np.sum(ancDoc[doc]**2))

print("ANC Done!")



queries = pd.read_csv("./Data/queries.csv")
queries = queries['query']
stop_words = set(stopwords.words('english'))
punc_tokenizer = nltk.RegexpTokenizer(r"\w+")
for i in range(queries.size) :
    word_tokens = word_tokenize(queries[i])
    word_tokens = [w.lower() for w in word_tokens]
    removed_stopwords = [w for w in word_tokens if not w.lower() in stop_words] # Removing all the stopwords in the queries
    sent_removed = ' '.join(removed_stopwords)
    only_words = punc_tokenizer.tokenize(sent_removed)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in only_words])
    queries[i] = lemmatized_output

print("Query Reading Done!")

query_TermFrequency = defaultdict(dict)  #finding TF part of query #defaultdict allows for 2d (or 3d or wtv)keys 

for i in range(queries.size):
    for word in queries[i].split(" "):
        if((i in query_TermFrequency)&(word in query_TermFrequency[i])):
                query_TermFrequency[i][word]+=1
        else:
                query_TermFrequency[i][word]=1

print("Query TF Done!")

# TF-IDF of query (ltc)
ltc_query = dict()
for query in query_TermFrequency:
    ltc_query[query] = np.zeros(len(keynum))
    for key in query_TermFrequency[query]:
        ltc_query[query][keynum[key]] = query_TermFrequency[query][key]
        if(ltc_query[query][keynum[key]] > 0):
            ltc_query[query][keynum[key]] = 1 + math.log10(ltc_query[query][keynum[key]])
    for key in DocumentFrequency:
        ltc_query[query][keynum[key]]*=(math.log10(len(processed_corpus)/DocumentFrequency[key]))
    ltc_query[query] /= np.sqrt(np.sum(ltc_query[query]**2))

print("LTC Done!")
    
# TF-IDF of query (lpc)
lpc_query = dict()
for query in query_TermFrequency:
    lpc_query[query] = np.zeros(len(keynum))
    for key in query_TermFrequency[query]:
        lpc_query[query][keynum[key]] = query_TermFrequency[query][key]
        if(lpc_query[query][keynum[key]] > 0):
            lpc_query[query][keynum[key]] = 1 + math.log10(lpc_query[query][keynum[key]])
    for key in DocumentFrequency:
        lpc_query[query][keynum[key]]*=max(0,(math.log10((len(processed_corpus)-DocumentFrequency[key])/DocumentFrequency[key])))
    lpc_query[query] /= np.sqrt(np.sum(lpc_query[query]**2))

print("LPC Done!")
    
# TF-IDF of query (apc)
apc_query = dict()
for query in query_TermFrequency:
    apc_query[query] = np.zeros(len(keynum))
    for key in query_TermFrequency[query]:
        apc_query[query][keynum[key]] = query_TermFrequency[query][key]
    apc_query[query] = 0.5 + 0.5*apc_query[query]/np.max(apc_query[query])
    for key in keynum:
        apc_query[query][keynum[key]]*=max(0,(math.log10((len(processed_corpus)-DocumentFrequency[key])/DocumentFrequency[key])))
    apc_query[query] /= np.sqrt(np.sum(apc_query[query]**2))

print("APC Done!")


def cos_sim(a,b):
    return dot(a, b)

def WriteDictToFile(my_dict,csv_file):
    try:
        with open(csv_file, 'w') as f:
            f.write('{0},{1}\n'.format('query id', 'document id'))
            [f.write('{0},{1}\n'.format(key, value)) for key, value in (my_dict.items())]
    except IOError:
        print("I/O error")

#lnc.ltc
rank=dict()
for query_id in range(queries.size):
    rank_list=list()
    for doc_id in lncDoc:
        rank_list.append([cos_sim(lncDoc[doc_id],ltc_query[query_id]),doc_id])
    rank_list.sort(reverse=True)
    rank[query_id]=' '.join([x[1] for x in rank_list[:50]])
WriteDictToFile(rank,"Assignment2_9_ranked_list_A.csv")

print("Writing File Done!")

#lnc.lpc
rank=dict()
for query_id in range(queries.size):
    rank_list=list()
    for doc_id in lncDoc:
        rank_list.append([cos_sim(lncDoc[doc_id],lpc_query[query_id]),doc_id])
    rank_list.sort(reverse=True)
    rank[query_id]=' '.join([x[1] for x in rank_list[:50]])
WriteDictToFile(rank,"Assignment2_9_ranked_list_B.csv")

print("Writing File Done!")

    
#anc.apc
rank=dict()
for query_id in range(queries.size):
    rank_list=list()
    for doc_id in ancDoc:
        rank_list.append([cos_sim(ancDoc[doc_id],apc_query[query_id]),doc_id])
    rank_list.sort(reverse=True)
    rank[query_id]=' '.join([x[1] for x in rank_list[:50]])
WriteDictToFile(rank,"Assignment2_9_ranked_list_C.csv")

print("Writing File Done!")











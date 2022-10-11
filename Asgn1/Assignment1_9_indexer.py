import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import OrderedDict

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


# Taking folder name as input using command line
path_to_json = sys.argv[1]
if not os.path.exists(path_to_json):
    print("Error : Folder doesn't Exist ")
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
idmap = pd.read_csv("./Data/id_mapping.csv")

for index, row in idmap.iterrows():
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
    removed_stopwords = [
        w for w in word_tokens if not w.lower() in stop_words
    ]  # Removing all the stopwords in the queries
    sent_removed = " ".join(removed_stopwords)
    only_words = punc_tokenizer.tokenize(sent_removed)
    lemmatized_output = " ".join([lemmatizer.lemmatize(w) for w in only_words])
    processed_corpus[doc] = lemmatized_output

# building inverted index
inverted_index = dict()
for doc in processed_corpus:
    unique_words = set()
    for word in processed_corpus[doc].split(" "):
        if word not in unique_words:
            if word in inverted_index:
                inverted_index[word].append(doc)
            else:
                inverted_index[word] = [doc]
        unique_words.add(word)



inverted_index = OrderedDict(sorted(inverted_index.items()))
for keys in inverted_index:
    inverted_index[keys].sort()

# convert the dictionary to pickle
pick_path = "./model_queries_9.bin"
with open(pick_path, "wb") as pick:
    pickle.dump(inverted_index, pick)
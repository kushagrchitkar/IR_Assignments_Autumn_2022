# 1B: Parser to process queries using Natural Language Processing techniques

import pandas as pd
import numpy as np
import os
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("omw-1.4")
lemmatizer = WordNetLemmatizer()
punc_tokenizer = nltk.RegexpTokenizer(r"\w+")
stop_words = set(stopwords.words("english"))

# Taking file name as input using command line
fn = sys.argv[1]
if not os.path.exists(fn):
    print("Error : File doesn't Exist ")
    sys.exit()

# Reading the CSV file and extracting the query column
query = pd.read_csv(fn)
query = query["query"]

# Set of all stop words
stop_words = set(stopwords.words("english"))

# Punctuation tokenizer
punc_tokenizer = nltk.RegexpTokenizer(r"\w+")

""" 
For every query:
1. We generate the word tokens
2. We remove all the stop words from the query
3. We generate the words remaining and lemmatize that to generate our final processed query
"""
for i in range(query.size):
    word_tokens = word_tokenize(query[i])  # 1
    removed_stopwords = [w for w in word_tokens if not w.lower() in stop_words]  # 2
    sent_removed = " ".join(removed_stopwords)  # 3
    only_words = punc_tokenizer.tokenize(sent_removed)  # 3
    lemmatized_output = " ".join([lemmatizer.lemmatize(w) for w in only_words])  # 3
    query[i] = lemmatized_output  # 3

# generating an index column
vals = np.vstack((np.arange(40), np.array(query))).T

# saving the file
np.savetxt("queries_9.txt", vals, delimiter=", ", newline="\n", fmt="%s")

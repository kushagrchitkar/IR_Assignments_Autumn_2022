import pandas as pd
import numpy as np
import os
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
punc_tokenizer = nltk.RegexpTokenizer(r"\w+")
stop_words = set(stopwords.words('english'))

fn = sys.argv[1]
if not os.path.exists(fn):
    print("Error : File doesn't Exist ")
    sys.exit()

query = pd.read_csv(fn)
query = query['query']
stop_words = set(stopwords.words('english'))
punc_tokenizer = nltk.RegexpTokenizer(r"\w+")
for i in range(query.size) :
    word_tokens = word_tokenize(query[i])
    removed_stopwords = [w for w in word_tokens if not w.lower() in stop_words] # Removing all the stopwords in the queries
    sent_removed = ' '.join(removed_stopwords)
    only_words = punc_tokenizer.tokenize(sent_removed)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in only_words])
    query[i] = lemmatized_output

vals = np.vstack((np.arange(40), np.array(query))).T
np.savetxt("queries_9.txt", vals, delimiter=", ", newline = "\n", fmt="%s")
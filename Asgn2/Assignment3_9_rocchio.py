import json
import os
import sys
import pickle
import pandas as pd
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
path_to_json = sys.argv[1]
path_to_index = sys.argv[2]
path_to_gold_standard = sys.argv[3]
path_to_ranked_list = sys.argv[4]

if not os.path.exists(path_to_json):
    print("Error : Data Folder doesn't Exist ")
    sys.exit()

if not os.path.exists(path_to_index):
    print("Error : Model doesn't Exist ")
    sys.exit()

if not os.path.exists(path_to_gold_standard):
    print("Error : qrel doc doesn't Exist ")
    sys.exit()

if not os.path.exists(path_to_ranked_list):
    print("Error : Ranked List A doesn't Exist ")
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
    if (row["paper_id"] in raw_dict):
        cord_to_paperId[row["cord_id"]] = row["paper_id"]
        paper_to_cordId[row["paper_id"]] = row["cord_id"]
        if row["cord_id"] not in corpus:
            corpus[row["cord_id"]] = raw_dict[row["paper_id"]]
        else:
            corpus[row["cord_id"]] = corpus[row["cord_id"]] + \
                " "+raw_dict[row["paper_id"]]

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

del (corpus)
del (raw_dict)
del (cord_to_paperId)
del (paper_to_cordId)
gc.collect()

DocumentFrequency = dict()

for keys in InvertedIndex:
    if (len(InvertedIndex[keys]) > 2):
        DocumentFrequency[keys] = len(InvertedIndex[keys])


print("DF Done!")

# finding TF part  #defaultdict allows for 2d (or 3d or wtv)keys
TermFrequency = defaultdict(dict)

for doc in processed_corpus:
    for word in processed_corpus[doc].split(" "):
        if (word in DocumentFrequency):
            if ((doc in TermFrequency) & (word in TermFrequency[doc])):
                TermFrequency[doc][word] += 1
            else:
                TermFrequency[doc][word] = 1

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
        if (lncDoc[doc][keynum[key]] > 0):
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

queries = pd.read_csv("./Data/queries.csv")
queries = queries['query']
stop_words = set(stopwords.words('english'))
punc_tokenizer = nltk.RegexpTokenizer(r"\w+")
for i in range(queries.size):
    word_tokens = word_tokenize(queries[i])
    word_tokens = [w.lower() for w in word_tokens]
    # Removing all the stopwords in the queries
    removed_stopwords = [w for w in word_tokens if not w.lower() in stop_words]
    sent_removed = ' '.join(removed_stopwords)
    only_words = punc_tokenizer.tokenize(sent_removed)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in only_words])
    queries[i] = lemmatized_output

print("Query Reading Done!")

# finding TF part of query #defaultdict allows for 2d (or 3d or wtv)keys
query_TermFrequency = defaultdict(dict)

for i in range(queries.size):
    for word in queries[i].split(" "):
        if (word in keynum):
            if ((i in query_TermFrequency) & (word in query_TermFrequency[i])):
                query_TermFrequency[i][word] += 1
            else:
                query_TermFrequency[i][word] = 1

print("Query TF Done!")

# TF-IDF of query (ltc)
ltc_query = dict()
for query in query_TermFrequency:
    ltc_query[query] = np.zeros(len(keynum))
    for key in query_TermFrequency[query]:
        ltc_query[query][keynum[key]] = query_TermFrequency[query][key]
        if (ltc_query[query][keynum[key]] > 0):
            ltc_query[query][keynum[key]] = 1 + \
                math.log10(ltc_query[query][keynum[key]])
    for key in DocumentFrequency:
        ltc_query[query][keynum[key]
                         ] *= (math.log10(len(processed_corpus)/DocumentFrequency[key]))
    ltc_query[query] /= np.sqrt(np.sum(ltc_query[query]**2))

print("LTC Done!")


def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))


def WriteDictToFile(my_dict, csv_file):
    try:
        with open(csv_file, 'w') as f:
            f.write('{0},{1}\n'.format('query id', 'document id'))
            [f.write('{0},{1}\n'.format(key, value))
             for key, value in (my_dict.items())]
    except IOError:
        print("I/O error")


# lnc.ltc
rank = dict()
for query_id in range(queries.size):
    rank_list = list()
    print(query_id)
    for doc_id in lncDoc:
        rank_list.append(
            [cos_sim(lncDoc[doc_id], ltc_query[query_id]), doc_id])
    rank_list.sort(reverse=True)
    rank[query_id] = ' '.join([x[1] for x in rank_list[:50]])

gold_standard = pd.read_csv(path_to_gold_standard)
gold_doc_list = gold_standard[gold_standard['judgement'] == 2]['cord-id']

relevant_docs = dict()
for val in gold_doc_list:
    relevant_docs[val] = True

del (gold_standard)
del (gold_doc_list)
gc.collect()

alphas = np.array([1, 0.5, 1])
betas = np.array([0.5, 0.5, .5])
gammas = np.array([0.5, 0.5, 0])


# df_q is the gold standard document frequency and rank is dictionary 0 to 35 each i with ranking of top 20 docs
def calcNCDG20_map20(rank, df_q):
    #############################
    #  Define the NCDG and MAP function here #
    #############################
    ap10 = []
    ap20 = []
    ndcg10 = []
    ndcg20 = []
    for k in range(1, len(set(df_q['topic-id']))+1):
        df_q_0 = df_q[df_q['topic-id'] == k]
        q_0 = {}
        for i in range(df_q_0.index[0], df_q_0.index[0] + len(df_q_0)):
            if df_q_0['cord-id'][i] in q_0:
                print("found it")
                if q_0[df_q_0['cord-id'][i]][0] >= df_q_0['iteration'][i]:
                    pass
                else:
                    q_0[df_q_0['cord-id'][i]] = [df_q_0['iteration']
                                                 [i], df_q_0['judgement'][i]]
            else:
                q_0[df_q_0['cord-id'][i]] = [df_q_0['iteration']
                                             [i], df_q_0['judgement'][i]]

        rel_docs = []
        for x in q_0:
            if q_0[x][1] == 1 or q_0[x][1] == 2:
                rel_docs.append(x)

        ret_docs = rank[k-1]

        precision_index = []
        for i in range(len(ret_docs[:10])):
            if ret_docs[i] in rel_docs:
                precision_index.append(i+1)
        sum_10 = 0
        ap_10 = 0
        for i in range(len(precision_index)):
            sum_10 += (i+1)/precision_index[i]

            ap_10 = sum_10/len(precision_index)

        ap10.append(ap_10)

        precision_index = []
        for i in range(len(ret_docs)):
            if ret_docs[i] in rel_docs:
                precision_index.append(i+1)
        sum_20 = 0
        ap_20 = 0
        for i in range(len(precision_index)):
            sum_20 += (i+1)/precision_index[i]
            ap_20 = sum_20/len(precision_index)

        ap20.append(ap_20)

        dcg = []
        for doc in ret_docs:
            if doc not in q_0:
                relevance = 0
            else:
                relevance = q_0[doc][1]
            dcg.append(relevance)

        ideal_dcg = sorted(dcg, reverse=True)

        for i in range(1, len(dcg)):
            dcg[i] = dcg[i]/math.log(i+1, 2)
        for i in range(1, len(dcg)):
            dcg[i] = dcg[i] + dcg[i-1]

        for i in range(1, len(ideal_dcg)):
            ideal_dcg[i] = ideal_dcg[i]/math.log(i+1, 2)
        for i in range(1, len(ideal_dcg)):
            ideal_dcg[i] = ideal_dcg[i] + ideal_dcg[i-1]

        print(k)
        try:
            ndcg = [dcg[i]/ideal_dcg[i] for i in range(len(dcg))]
        except:
            ndcg = [0]
        ndcg20.append(ndcg[-1])

        dcg = []
        for doc in ret_docs[:10]:
            if doc not in q_0:
                relevance = 0
            else:
                relevance = q_0[doc][1]
            dcg.append(relevance)

        ideal_dcg = sorted(dcg, reverse=True)

        for i in range(1, len(dcg)):
            dcg[i] = dcg[i]/math.log(i+1, 2)
        for i in range(1, len(dcg)):
            dcg[i] = dcg[i] + dcg[i-1]

        for i in range(1, len(ideal_dcg)):
            ideal_dcg[i] = ideal_dcg[i]/math.log(i+1, 2)
        for i in range(1, len(ideal_dcg)):
            ideal_dcg[i] = ideal_dcg[i] + ideal_dcg[i-1]

        try:
            ndcg = [dcg[i]/ideal_dcg[i] for i in range(len(dcg))]
        except:
            ndcg = [0]
        ndcg10.append(ndcg[-1])
    ndcgRet = np.average([x for x in ndcg20 if str(x) != 'nan'])
    map20Ret = np.average([x for x in ap20 if str(x) != 'nan'])
    return ndcgRet, map20Ret


df_q = pd.read_csv(path_to_gold_standard)
output_to_csv = np.array([[]])
for i in range(len(alphas)):
    each_ltc_query = dict()
    for j in range(len(ltc_query)):
        each_ltc_query[j] = np.zeros(len(ltc_query[0]))
        each_ltc_query[j] = each_ltc_query[j] + ltc_query[j]

    num_rel_docs = 0
    d_rel = np.zeros(len(ltc_query[0]))
    for j in relevant_docs:
        num_rel_docs += 1
        if j in lncDoc:
            d_rel = d_rel + lncDoc[j]

    num_non_rel_docs = 0
    d_non_rel = np.zeros(len(ltc_query[0]))
    for key in TermFrequency:
        if key not in relevant_docs:
            num_non_rel_docs += 1
            if key in lncDoc:
                d_non_rel = d_non_rel + lncDoc[key]
    for key in each_ltc_query:
        qm = alphas[i]*each_ltc_query[key] + betas[i]*d_rel / \
            num_rel_docs - gammas[i]*d_non_rel/num_non_rel_docs
        each_ltc_query[key] = qm
    # lnc.ltc
    rank = dict()
    for query_id in range(queries.size):
        rank_list = list()
        for doc_id in lncDoc:
            rank_list.append(
                [cos_sim(lncDoc[doc_id], each_ltc_query[query_id]), doc_id])
        rank_list.sort(reverse=True)
        rank[query_id] = [x[1] for x in rank_list[:20]]
    ncdg20, map20 = calcNCDG20_map20(rank, df_q)
    output_to_csv = np.append(output_to_csv, np.array(
        [alphas[i], betas[i], gammas[i], ncdg20, map20]))
    print(alphas[i], ',', betas[i], ',', gammas[i], ncdg20, map20)
output_to_csv = output_to_csv.reshape(len(alphas), 5)
pd.DataFrame(output_to_csv).to_csv('Assignment3_9_rocchio_RF_metrics.csv',
                                   index_label="Index", header=['alpha', 'beta', 'gamma', 'mAP@20', 'NDCG@20'])

print("Relevance Feedback Done")

df_ranked = pd.read_csv(path_to_ranked_list)

pseudo_rel_docs = dict()
for i in range(len(df_ranked)):
    for word in df_ranked['document id'][i].split(' ')[:10]:
        pseudo_rel_docs[word] = True

output_to_csv = np.array([[]])
for i in range(len(alphas)):
    each_ltc_query = dict()
    for j in range(len(ltc_query)):
        each_ltc_query[j] = np.zeros(len(ltc_query[0]))
        each_ltc_query[j] = each_ltc_query[j] + ltc_query[j]

    num_rel_docs = 0
    d_rel = np.zeros(len(ltc_query[0]))
    for j in pseudo_rel_docs:
        num_rel_docs += 1
        if j in lncDoc:
            d_rel = d_rel + lncDoc[j]

    for key in each_ltc_query:
        qm = alphas[i]*each_ltc_query[key] + betas[i]*d_rel/num_rel_docs
        each_ltc_query[key] = qm
    # lnc.ltc
    rank = dict()
    for query_id in range(queries.size):
        rank_list = list()
        print(query_id)
        for doc_id in lncDoc:
            rank_list.append(
                [cos_sim(lncDoc[doc_id], each_ltc_query[query_id]), doc_id])
        rank_list.sort(reverse=True)
        rank[query_id] = [x[1] for x in rank_list[:20]]
    ncdg20, map20 = calcNCDG20_map20(rank, df_q)
    output_to_csv = np.append(output_to_csv, np.array(
        [alphas[i], betas[i], gammas[i], ncdg20, map20]))
    print(alphas[i], ',', betas[i], ',', gammas[i], ncdg20, map20)
output_to_csv = output_to_csv.reshape(len(alphas), 5)
pd.DataFrame(output_to_csv).to_csv('Assignment3_9_rocchio_PsRF_metrics.csv',
                                   index_label="Index", header=['alpha', 'beta', 'gamma', 'mAP@20', 'NDCG@20'])

print("Pseudo Relevance Feedback Done")

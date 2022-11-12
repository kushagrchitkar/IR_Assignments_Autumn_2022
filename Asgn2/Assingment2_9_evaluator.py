import pandas as pd
import numpy as np
import math
import os
import sys

#reading qrels and ranked list files
path_to_goldstandard = sys.argv[1]
path_to_list = sys.argv[2]

if not os.path.exists(path_to_goldstandard):
    print("Error : Goldstandard doesn't Exist ")
    sys.exit()

if not os.path.exists(path_to_list):
    print("Error : List doesn't Exist ")
    sys.exit()

df = pd.read_csv(path_to_list)
df_q = pd.read_csv(path_to_goldstandard)


#determining which results file to write to
if '_A' in path_to_list:
    s = 'A'
elif '_B' in path_to_list:
    s = 'B'
elif '_C' in path_to_list:
    s = 'C'

ap10 = []
ap20 = []
ndcg10 = []
ndcg20 = []

for k in range(1, len(set(df_q['topic-id']))+1):
  df_q_0 = df_q[df_q['topic-id'] == k]

  #setting iteration priority order
  q_0 = {}
  for i in range(df_q_0.index[0], df_q_0.index[0] + len(df_q_0)):
    if df_q_0['cord-id'][i] in q_0:
      print("found it")
      if q_0[df_q_0['cord-id'][i]][0] >= df_q_0['iteration'][i]:
        pass
      else:
        q_0[df_q_0['cord-id'][i]] = [df_q_0['iteration'][i], df_q_0['judgement'][i]]
    else:
      q_0[df_q_0['cord-id'][i]] = [df_q_0['iteration'][i], df_q_0['judgement'][i]]

  rel_docs = []
  for x in q_0:
    if q_0[x][1] == 1 or q_0[x][1] == 2:
      rel_docs.append(x)
  
  ret_docs = df.loc[k-1]['document id']
  ret_docs = ret_docs.split(' ')[:20]

  #calculating average precision for first 10 ranked documents
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

  #calculating average precision for first 20 ranked documents
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

  #calculating ndcg for first 10 ranked documents
  dcg = []
  for doc in ret_docs:
    if doc not in q_0:
      relevance = 0
    else:
      relevance = q_0[doc][1]
    dcg.append(relevance)

  ideal_dcg = sorted(dcg, reverse=True)

  for i in range(1,len(dcg)):
    dcg[i] = dcg[i]/math.log(i+1,2)
  for i in range(1,len(dcg)):
    dcg[i] = dcg[i] + dcg[i-1]
  
  for i in range(1,len(ideal_dcg)):
    ideal_dcg[i] = ideal_dcg[i]/math.log(i+1,2)
  for i in range(1,len(ideal_dcg)):
    ideal_dcg[i] = ideal_dcg[i] + ideal_dcg[i-1]

  
  try:
    ndcg = [dcg[i]/ideal_dcg[i] for i in range(len(dcg))]
  except:
    ndcg = [0]
  ndcg20.append(ndcg[-1])

  #calculating ndcg for first 20 ranked documents

  dcg = []
  for doc in ret_docs[:10]:
    if doc not in q_0:
      relevance = 0
    else:
      relevance = q_0[doc][1]
    dcg.append(relevance)

  ideal_dcg = sorted(dcg, reverse=True)

  for i in range(1,len(dcg)):
    dcg[i] = dcg[i]/math.log(i+1,2)
  for i in range(1,len(dcg)):
    dcg[i] = dcg[i] + dcg[i-1]
  
  for i in range(1,len(ideal_dcg)):
    ideal_dcg[i] = ideal_dcg[i]/math.log(i+1,2)
  for i in range(1,len(ideal_dcg)):
    ideal_dcg[i] = ideal_dcg[i] + ideal_dcg[i-1]

  try:
    ndcg = [dcg[i]/ideal_dcg[i] for i in range(len(dcg))]
  except:
    ndcg = [0]
  ndcg10.append(ndcg[-1])

#calculating map values and averNDCG values
def av(arr):
  return sum(arr)/len(arr)
def remove_nan(arr):
  for i in range(len(arr)):
    if np.isnan(arr[i]):
      arr[i] = 0
  return arr

remove_nan(ndcg10)
remove_nan(ndcg20)

map_10 = av(ap10)
map_20 = av(ap20)
averndcg10 = av(ndcg10)
averndcg20 = av(ndcg20)

#writing to txt file

with open("Assignment2_9_metrics_" + s + ".txt", "w") as txt_file:
  txt_file.write("Average Precision(AP) @ 10 values for all queries \n")
  txt_file.write("\n")
  for i in range(len(ap10)):
    txt_file.write(str(i) +" : " + str(ap10[i]) + "\n")

  txt_file.write("\n")
  txt_file.write("Average Precision(AP) @ 20 values for all queries \n")
  txt_file.write("\n")
  for i in range(len(ap20)):
    txt_file.write(str(i) +" : " + str(ap20[i]) + "\n") 

  txt_file.write("\n")
  txt_file.write("NDCG @ 10 values for all queries \n")
  txt_file.write("\n")
  for i in range(len(ndcg10)):
    txt_file.write(str(i) +" : " + str(ndcg10[i]) + "\n") 

  txt_file.write("\n")
  txt_file.write("NDCG @ 20 values for all queries \n")
  txt_file.write("\n")
  for i in range(len(ndcg20)):
    txt_file.write(str(i) +" : " + str(ndcg20[i]) + "\n") 

  txt_file.write("\n")
  txt_file.write("Mean Average Precision (MAP@10) : " + str(map_10) + "\n")
  txt_file.write("Mean Average Precision (MAP@20) : " + str(map_20) + "\n")
  txt_file.write("Average NDCG (averNDCG@10) : " + str(averndcg10) + "\n")
  txt_file.write("Average NDCG (averNDCG@20) : " + str(averndcg20) + "\n")

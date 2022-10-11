import pickle
import os
import sys

path_to_model = sys.argv[1]
path_to_queries = sys.argv[2]

if not os.path.exists(path_to_model):
    print("Error : Model doesn't Exist ")
    sys.exit()

if not os.path.exists(path_to_queries):
    print("Error : Queries doesn't Exist ")
    sys.exit()

f = open(path_to_model, "rb")
m_query = pickle.load(f)


query_dict = {}
temp = open(path_to_queries,'r').read().splitlines()

for i in range(len(temp)):
  temp[i] = temp[i].split(", ")
  query_dict[int(temp[i][0])] = temp[i][1]

def merge_arr(l1,l2):
  final_l = []
  i = j = 0
  while i != len(l1) and j != len(l2):
    if l1[i] == l2[j]:
      final_l.append(l1[i])
      i += 1
      j += 1
    else:
      if l1[i] < l2[j]:
        i += 1
      else:
        j += 1
  return final_l

def sort_by_len(l):
  freq = [len(m_query[i]) for i in l]
  sorted_l = [x for _, x in sorted(zip(freq, l))]
  return sorted_l

def multiple_merge(arr):
  i = 1
  l = m_query[arr[i-1]]
  while i != len(arr):
    l = merge_arr(m_query[arr[i]], l)
    i += 1
  return l

final_docs_retrieved = []
print("Retrieving documents for queries")
for i in range(len(query_dict)):
  a = query_dict[i].split(" ")
  a = multiple_merge(sort_by_len(a))
  final_docs_retrieved.append(a)

print("Done!")
print("Writing documents to results.txt")

with open("results.txt", "w") as txt_file:
    for i in range(len(final_docs_retrieved)):
        txt_file.write(str(i) +" : " + " ".join(final_docs_retrieved[i]) + "\n")

print("Done! Please check results.txt file")
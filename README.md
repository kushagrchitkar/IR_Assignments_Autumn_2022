# Information Retrieval Course Repository - Autumn 2022

Welcome to the repository for the Information Retrieval course at the Indian Institute of Technology, Kharagpur, for the Autumn 2022 semester.

## Overview

This repository contains the code for assignments related to Information Retrieval. The primary focus is on building effective retrieval systems for documents using different techniques.

## Assignment Highlights

### Assignment 1: Boolean Retrieval with Inverted Index

In the initial assignment, we built a simple inverted index. The Boolean Retrieval system treats each query as an AND of individual words, utilizing this encoding to retrieve all relevant documents.

### Assignment 2: Ranked Retrieval with TF-IDF

For the subsequent assignment, we advanced to a tf-idf based ranked retrieval system. The tf-idf weight, representing the term frequency of a term in a document, was used for each term. This enhanced system aims to answer free-text queries.

### Evaluation Metrics

To assess the performance of our information retrieval system, we gathered the top 20 documents and compared them to the gold standard, provided in `qrels.csv`. Evaluation metrics such as Average Precision and Normalized Discounted Cumulative Gain were employed.

### Assignments 3: Relevance Feedback

In the final phase, relevance feedback methods were applied to enhance system recall. For relevance feedback, a gold standard relevance judgment score of 2 was considered as relevant, and the rest as non-relevant. Pseudo Relevance Feedback was implemented by deeming the top 10 ranked documents as relevant, and then modifying the query vector using Rocchio's algorithm.

## Code and Results

All the code and results related to the assignments have been uploaded in this repository. You can explore the respective folders for detailed implementations, code documentation, and evaluation results.

## How to Use

Feel free to clone or fork this repository to explore the code, replicate the experiments, or adapt it for your own learning purposes.

We appreciate your interest and hope you find the contents of this repository valuable for your understanding of Information Retrieval.

Best Regards,
Kushagra Chitkara

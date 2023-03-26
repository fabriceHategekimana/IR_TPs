

import os
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

# Load the NLTK stop words 
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))



def read_files(folder_path, num_articles=15):
    file_names = os.listdir(folder_path)
    # Filter only the text files
    text_files = [file_name for file_name in file_names if file_name.endswith('.txt')]
    docs = []
    filenames = []  
    for file_name in text_files[:num_articles]:
        with open(os.path.join(folder_path, file_name), 'r', encoding='windows-1252') as file:
            doc = file.readlines()
            docs.append(doc)
            filenames.append(file_name)  
    return docs, filenames  



def clean_raw_text(docs):
    
    punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    digits = '0123456789'
    
    for doc_num, doc in enumerate(docs):
        # doc is a list of sentences
        for sent_num, sent in enumerate(doc):
            # Lowercase
            sent = sent.lower()
            
            # Removing the punctuations and numbers
            sent = "".join([char for char in sent if char not in punctuations + digits])
            docs[doc_num][sent_num] = sent
            
    return docs

def preprocess(folder_path):
    
    # Read files from folder
    docs, filenames = read_files(folder_path)   
       
    # Clean raw text
    cleaned_docs = clean_raw_text(docs)

    # Tokenize documents
    docs_tokenized = []
    for doc in cleaned_docs:
        tokenized_doc = [nltk.word_tokenize(sent) for sent in doc]
        docs_tokenized.append(tokenized_doc)

    # Stemming
    ps = PorterStemmer()
    docs_stemmed = []
    for doc in docs_tokenized:
        stemmed_doc = [ps.stem(token) for sent in doc for token in sent]
        docs_stemmed.append(stemmed_doc)

    return docs_stemmed, filenames



def tokenize_queries(queries):
    
    queries = clean_raw_text([queries])[0]
    
    queries_tokenized = []
    
    for query_num, query in enumerate(queries):
        # query is a string (sentence)
        queries_tokenized.append(nltk.word_tokenize(query))  
        
    return queries_tokenized


def compute_tf_idf(tokenized_corpus, remove_stop_words=False, top_p_stems=None):
    print(f"Stop words removal: {remove_stop_words}")
    # Remove stop words if specified
    if remove_stop_words:
        tokenized_corpus = [[token for token in doc if token not in stop_words] for doc in tokenized_corpus]

    # Compute term frequency
    unique_words = list(set([word for doc in tokenized_corpus for word in doc]))
    num_docs = len(tokenized_corpus)
    num_words = len(unique_words)
    tf_matrix = np.zeros((num_docs, num_words))

    for i, doc in enumerate(tokenized_corpus):
        for word in doc:
            j = unique_words.index(word)
            tf_matrix[i, j] += 1

    # Compute inverse document frequency
    idf = np.zeros(num_words)
    for i, word in enumerate(unique_words):
        docs_containing_word = sum([1 for doc in tokenized_corpus if word in doc])
        idf[i] = np.log(num_docs / docs_containing_word)

    # Compute tf-idf
    tf_idf_matrix = tf_matrix * idf

    # Create pandas DataFrames
    tf_df = pd.DataFrame(tf_matrix, columns=unique_words)
    tf_idf_df = pd.DataFrame(tf_idf_matrix, columns=unique_words)
    if top_p_stems:
        top_p_tfidf = tf_idf_df.apply(lambda x: x.nlargest(top_p_stems).index.tolist(), axis=1)
        return tf_df, tf_idf_df, top_p_tfidf
    else:
        return tf_df, tf_idf_df
    
    
def build_lsi(tf_idf_matrix, k=2):
    # Compute the Singular Value Decomposition (SVD)
    U, s, Vt = np.linalg.svd(tf_idf_matrix, full_matrices=False)

    # Reduce the dimensionality of the SVD matrices
    U_k = U[:, :k]
    s_k = np.diag(s[:k])
    Vt_k = Vt[:k, :]

    # Compute the reduced document representation
    lsi_docs = np.dot(U_k, s_k)

    return lsi_docs, U_k, s_k, Vt_k

def query_lsi(queries_tokenized, U_k, s_k, Vt_k, unique_words):
    # Convert unique_words to a list
    unique_words = unique_words.tolist()

    # Create a term-query matrix
    tq_matrix = np.zeros((len(unique_words), len(queries_tokenized)))
    for j, query in enumerate(queries_tokenized):
        for word in query:
            if word in unique_words:
                i = unique_words.index(word)
                tq_matrix[i, j] += 1

    # Compute the reduced query representation
    lsi_queries = np.dot(np.dot(np.linalg.inv(s_k), Vt_k), tq_matrix).T

    return lsi_queries



def rank_documents(lsi_docs, lsi_queries):
    # Compute the cosine similarity between queries and documents
    cosine_similarities = cosine_similarity(lsi_queries, lsi_docs)

    # Rank the documents and their similarity scores for each query
    sorted_indices = np.argsort(cosine_similarities, axis=1)[:, ::-1]
    sorted_similarities = np.sort(cosine_similarities, axis=1)[:, ::-1]

    return sorted_indices, sorted_similarities

def main():
    folder_path = './nasa/'
    REMOVE_STOP_WORDS = True
    processed_docs, filenames = preprocess(folder_path)
    # q1: 01995.txt
    # q2: 01995.txt
    # q3: 04495.txt 
    QUERIES = ['integrated process to accomplish design',
               'smooth transfers of information',
               'sound generation by flows']

    queries_tokenized = tokenize_queries(QUERIES)
    tf_matrix, tf_idf_matrix, _ = compute_tf_idf(processed_docs, remove_stop_words=REMOVE_STOP_WORDS , top_p_stems=50)

    # Build the LSI model
    lsi_docs, U_k, s_k, Vt_k = build_lsi(tf_idf_matrix, k=2)

    # Calculate the LSI representation of the queries
    lsi_queries = query_lsi(queries_tokenized, U_k, s_k, Vt_k, tf_idf_matrix.columns)

    # Rank the documents for each query
    rankings, similarity_scores = rank_documents(lsi_docs, lsi_queries)

    # Create a DataFrame with rankings and similarity scores
    ranking_df = pd.DataFrame(rankings, index=QUERIES, columns=[f"Rank_{i+1}" for i in range(rankings.shape[1])]).transpose()
    for i, col in enumerate(ranking_df.columns):
        ranking_df[col] = ranking_df[col].apply(lambda x: f"{x} [{filenames[x]}]")
    similarity_df = pd.DataFrame(similarity_scores, index=QUERIES, columns=[f"Score_{i+1}" for i in range(similarity_scores.shape[1])]).transpose()

    # Print the DataFrames
    print("Rankings:")
    print(ranking_df)
    print("\nSimilarity Scores:")
    print(similarity_df)
    
main()
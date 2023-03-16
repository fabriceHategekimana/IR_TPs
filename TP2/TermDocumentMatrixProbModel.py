import numpy as np
import pandas as pd
from pipe import select
from src.module import file_name_to_text
from src.module import list_files_with_extention
from src.module import tokenize_doc
from src.module import term_frequence
from src.module import stem


def extract_text(article):
    file_name_to_text(article, "articles/")


def tf_computation(word, doc):
    freq_term = term_frequence(doc)
    max_count = freq_term.get(freq_term.max())
    word_count = doc.count(word)
    return word_count/max_count


def tf(word, docs):
    return [tf_computation(word, doc) for doc in docs]


def doc_occurence(term, docs):
    return sum(list(map(lambda doc: term in doc, docs)))


def idf(word, docs):
    N = len(docs)
    occ = doc_occurence(word, docs)
    return np.log(N/occ)


def tf_idf(word, docs):
    tfs = tf(word, docs)
    idfs = idf(word, docs)
    return [tf*idfs for tf in tfs]


def to_df_matrix(function, docs):
    unique_stems = list({stem for element in docs for stem in element})
    df_stems = pd.DataFrame({"stems": unique_stems})
    matrix = np.array([function(stem, docs) for stem in unique_stems])
    columns = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15"]
    df = pd.DataFrame(matrix, columns=columns)
    return pd.concat([df_stems, df], axis=1)


def get_top_p(df, p, name="sum"):
    df[name] = df.iloc[:, 1:16].sum(axis="columns")
    df_sorted = df.sort_values(by=[name], ascending=False)[:p]
    return df_sorted.reset_index(drop=True)


def boolean_normalization(df):
    df.iloc[:, 1:17] = df.iloc[:, 1:17].applymap(lambda x: 1 if x > 0 else 0)
    return df


def createTermDocumentMatrix(directory, number=15, kind="boolean", algo="tf", top_p=50):
    # select 15 first files
    selected_articles = list_files_with_extention(".txt", directory)[:number]
    doc_stems = list(selected_articles
                        | select(extract_text)
                        | select(tokenize_doc)
                        | select(stem)
                        )
    algo = tf if algo == "tf" else tf_idf
    df_matrix = to_df_matrix(algo, doc_stems)
    df_matrix = get_top_p(df_matrix, top_p)
    normalization = boolean_normalization
    return normalization(df_matrix.copy())


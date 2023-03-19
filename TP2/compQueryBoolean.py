import numpy as np


def compQueryBoolean(query, doc, top=5):
    res = boolean_query(query, doc)
    final = []
    for i, r in enumerate(res):
        final.append("d{i}")
    return final


def doc_contains(doc, vec_query):
    diff = doc-vec_query
    return not np.any([True if el < 0 else False for el in diff])


def boolean_query(query, df):
    terms = query.split(" and ")
    positions = [df[df["stems"] == term].index[0] for term in terms]
    row_number = len(df.index)
    vec_query = np.array([1 if p in positions else 0 for p in range(row_number)])
    return [doc_contains(df[f"d{i}"], vec_query) for i in range(1, 16)]

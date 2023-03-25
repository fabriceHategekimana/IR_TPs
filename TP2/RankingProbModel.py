import numpy as np


def cond_prob(ti, R, V, i):
    """
    Compute the probability for the R set
    V: set of the top r ranked documents
    Vi: subset of V containing the term i
    i: the index of term i
    R: set of document term
    """
    return (len(V[i])+0.5)/len(V)+1


def cond_prob_comp(ti, R, V, i, ni, N):
    """
    Compute the probability for the not R set
    V: set of the top r ranked documents
    Vi: subset of V containing the term i
    ni: number of documents where ti appear
    N: total Number of documents
    """
    return (ni-len(V[i])+0.5)/(N-len(V)+1)


def sim(dj, q, W, Q):
    """
    compute the similarity between a document and a query
    dj: The jth document
    q: the query in question
    W: the set of weights according to the documents
    Q: the set of weights according to the queries
    """
    term = W["term"]
    sims = []
    for i in len(term):
        wiq = Q[q][i]  # OK
        wij = W[dj][q]
        ti = term[i]
        p1 = cond_prob(ti, W)
        p2 = cond_prob_comp(ti, W)
        left = np.log(p1/1-p1)
        right = np.log(1-p2/p2)
        sims.append(wiq*wij*(left+right))
    return sims

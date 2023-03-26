# basic import
import os
import re
import nltk
import tarfile
import numpy as np
import pandas as pd
from pipe import select, where, traverse

# nltk, wordprocessig and matplotlib import
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
from fuzzywuzzy import fuzz, process
import matplotlib.pyplot as plt

# UNCOMMENT IF YOU ENCOUNTER AN ERROR WITH STOPWORDS OR PUNKT
# nltk.download("stopwords")
# nltk.download("punkt")


# reccurent types
# list of document that have multiple lines (= multiple str)
Sentences = list[str]
Tokens = list[str]


def extract_tar_file(path_to_file: str, path_to_extract: str = ".") -> None:
    """
    take a given path (path_to_file) and
    extract to a destination directory (path_to_extract)
    """
    path_to_file = os.path.join(path_to_file)
    path_to_extract = os.path.join(path_to_extract)

    with tarfile.open(path_to_file, "r:gz") as tar:
        tar.extractall(path_to_extract)


def list_files_with_extension(extension: str, target_dir: str = ".") -> Sentences:
    """
    list the file having a given extension
    and in a specific target_dir (target directory)
    """
    return [
            file for file in os.listdir(target_dir) if file.endswith(extension)
            ]


def file_names_to_text(list_of_files: Sentences, root_docs_path: str = ".") -> Sentences:
    """
    take a list of file names and return their content in plain string
    """
    def get_lines(file_name):
        f = open(
                os.path.join(root_docs_path, file_name),
                mode="r",
                encoding="windows-1252")
        res = f.readlines()
        f.close()
        return res
    return list(map(lambda x: get_lines(x), list_of_files))


def file_name_to_text(file_name: str, root_docs_path: str = "."):
    """
    take a file name and return it's content in plain string
    root_docs_path: folder to look for the file
    """
    f = open(root_docs_path+"/"+file_name, mode="r", encoding="windows-1252")
    lines = f.readlines()
    f.close()
    return lines


def clean_raw_text(doc: Sentences) -> Sentences:
    """
    remove unecessary numbers and characters
    in a set of documents
   """
    punctuations = r"""!()-[]{};:""\,<>./?@#$%^&*_~"""
    digits = "0123456789"

    # Removing the punctuations and numbers
    def remove_punctuations(x):
        return "".join(
                [char for char in x if char not in punctuations + digits])

    return list(doc
                | select(lambda x: x.lower())
                | select(lambda x: remove_punctuations(x))
                )


def get_stop_words(language="english") -> set[str]:
    """
    Get the list of stop words
    """
    return set(stopwords.words(language))


def tokenize_doc(doc: Sentences) -> Tokens:
    """
    each line into a list of words (=tokens)
    in a set of documents
    """
    doc = clean_raw_text(doc)

    def my_tokenization(sentence: str) -> list[str]:
        return nltk.word_tokenize(sentence)

    return list(doc
                | select(lambda x: my_tokenization(x))
                | where(lambda x: x != [])
                | traverse)


def stem(doc: Tokens) -> Tokens:
    """
    stems (reduce) words of the Docs (remove suffix)
    ex: talking -> talk
    """
    ps = PorterStemmer()
    return [ps.stem(word) for word in doc]


def lemmatize(docs):
    """
    find the base word with the knowledge of the language
    ex: ate -> eat
    """
    ps = WordNetLemmatizer()
    return list(map(lambda doc: [ps.lemmatize(word, "v") for word in doc], docs))


def generate_word_cloud(words: list[str]):
    """
    Generate a word cloud given a specific doc index
    """
    text = " ".join(words)
    wordcloud = WordCloud(
            width=800,
            height=800,
            background_color="white",
            min_font_size=10).generate(text)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


def fuzzy_search(word, docs):
    """
    Do a fuzzy search by words
    """
    choices = [item for sublist in docs for item in sublist]
    ratio_threshold = 70

    results = process.extract(word, choices, scorer=fuzz.partial_ratio)
    matches = [(r[0], r[1]) for r in results if r[1] >= ratio_threshold]
    print(f"The word '{word}' has the following fuzzy matches: {matches}")


def boolean_search(term, docs):
    """
    Do a boolean search with a given term
    the term can be a re regex
    """
    regex = re.compile(term)
    text = " ".join(docs[1])
    matches = regex.findall(text)
    print(f"The following words matched the search criteria: {matches}")


def term_frequence(tokens: Tokens) -> nltk.probability.FreqDist:
    sentences = " ".join(tokens)
    return FreqDist(word.lower() for word in word_tokenize(sentences))
    # use .most_common() to get the list of the most common tuples (term, count)


def tf(word, docs):
    def tf_computation(word, doc):
        freq_term = term_frequence(doc)
        max_count = freq_term.get(freq_term.max())
        word_count = doc.count(word)
        return word_count/max_count
    return [tf_computation(word, doc) for doc in docs]


def idf(word, docs):
    N = len(docs)
    def doc_occurence(term, docs):
        return sum(list(map(lambda doc: term in doc, docs)))
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
	df[name] = df.iloc[:,1:16].sum(axis="columns")
	df_sorted = df.sort_values(by=[name], ascending=False)[:p]
	return df_sorted.reset_index(drop=True)

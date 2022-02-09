import pandas as pd
import string
import spacy
import math
from nltk.stem.snowball import SnowballStemmer

# Preparation of data for processing
#=================================================
# 1  
def delete_non_literal(txt):
    ascii = (string.ascii_letters + " ")
    data1 = (c for c in txt if c in ascii)
    txt1 = "".join(data1)
    return ' '.join(txt1.split())


# 2
def tokenization(txt):
    return [str(tok) for tok in txt]


# 3
def delete_stopwords(tokenn):
    return [tok for tok in tokenn if not tok in spacy_stopwords]


# 4
def case_change(tokenn):
    return [tok.lower() for tok in tokenn]


# 5
def stemming(tokenn):
    return [stemmer.stem(tok) for tok in tokenn]


# 6
def lemmatization(txt):
    return [tok.lemma_ for tok in txt]


def all(txt):
    x = nlp(delete_non_literal(txt))
    a = tokenization(x)
    b = delete_stopwords(case_change(a))
    c = lemmatization(nlp(' '.join(b)))
    return stemming(c)
#=================================================

def Counter(txt):
    vocab = {}
    for token in txt:
        if token not in vocab.keys():
            vocab[token] = 1
        else:
            vocab[token] += 1
    return vocab


def tfidf_metric(corpus):
    def compute_tf(text):
        tf_text = Counter(text)
        for i in tf_text:
            tf_text[i] = tf_text[i] / float(len(text))
        return tf_text

    def compute_idf(word, corpus):
        return math.log(len(corpus) / sum([1.0 for i in corpus if word in i]))

    lst = []

    for text in corpus:
        tf_idf = {}
        tf = compute_tf(text)
        for word in tf:
            tf_idf[word] = tf[word] * compute_idf(word, corpus)
        lst.append(tf_idf)
    return lst

#=================================================
nlp = spacy.load('en_core_web_sm')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
stemmer = SnowballStemmer(language='english')

# Reading
#=================================================
df = pd.read_csv('data.csv', encoding='utf8')
df = df[["titles", "summary"]]
df = df[df["summary"].notna()]

# Preparation
#=================================================
arr = []
for i in df["summary"]:
    lst = all(i)
    arr.append(lst)
    
# TF-IDF vectorization    
#=================================================
tf_idf = tfidf_metric(arr)
with open('output.txt', 'w', encoding='utf-8') as fw:
    for i in range(len(tf_idf)):
        fw.write(str(df[["titles"]].values[i]) + " -> " + str(tf_idf[i]) + "\n")
print("Results in output.txt")
#=================================================

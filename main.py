# -*- coding:utf-8 -*-
u"""
Created on 07/04/15
by fccoelho
license: GPL V3 or Later
"""

__docformat__ = 'restructuredtext en'

import pandas as pd
import pymongo
from nltk.tokenize import sent_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
import nltk
from gensim.models.word2vec import Word2Vec
import logging
from string import punctuation
import re
import string

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                    level=logging.INFO)

nltk.download('stopwords')
sw = stopwords.words('portuguese') + list(string.punctuation)

pattern = re.compile(r"[^a-zA-Z\s]")

# Set values for various parameters
num_features = 300  # Word vector dimensionality
min_word_count = 40  # Minimum word count
num_workers = 8  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words


def querydb():
    con = pymongo.MongoClient('localhost', port=27000)
    db = con.MCDB
    col = db.articles
    for doc in col.find({"cleaned_text": {"$exists": True}}, {'cleaned_text': True, 'link': True, 'published': True}):
        yield doc


def get_phrases(doc):
    frases = sent_tokenize(doc['cleaned_text'])
    frases = [wordpunct_tokenize(frase.lower().strip().strip(punctuation)) for frase in frases if frase.strip()]
    return frases


def save_locally():
    con = pymongo.MongoClient('localhost', port=27017)
    con.drop_database("word2vec")
    col = con.word2vec.frases
    count = 1
    for doc in querydb():
        if doc['cleaned_text'] == "":
            continue
        doc['frases'] = get_phrases(doc)
        col.insert(doc)
        if count % 1000 == 0:
            print("saved {} documents".format(count))
        count += 1
    con.close()


def sentence_gen(limit=20e6):
    """
    Carrega as frases da coleção local para treinamento do modelo do W2v
    :return:
    """
    con = pymongo.MongoClient('localhost', port=27017)
    col = con.word2vec.frases
    for doc in col.find():
        for frase in doc['frases']:
            frase = [w.strip(punctuation) for w in frase if w not in sw]
            yield frase

def text_gen(limit=2e6):
    con = pymongo.MongoClient('localhost', port=27017)
    col = con.word2vec.frases
    for doc in col.find():
        text = pattern.sub("", doc['cleaned_text'])
        yield wordpunct_tokenize(text.lower())


def train_w2v_model(model_name="MediaCloud_w2v"):
    print("Training model...")
    model = Word2Vec(sentence_gen(), workers=num_workers, \
        size=num_features, min_count=min_word_count, \
        window=context)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model.save(model_name)
    print("Palavras mais similares a 'presidente':\n", model.most_similar("presidente"))

def train_d2v_model(model_name="MediaCloud_d2v"):
    model = Doc2Vec(sentence_gen(), size=100, window=8, min_count=45, workers=num_workers)
    model.save(model_name)
    print("Palavras mais similares a 'presidente':\n", model.most_similar("presidente"))

def train_w2v_model_per_article(model_name="MediaCloud_d2v"):
    model = Word2Vec(text_gen(10000), size=500, window=8, min_count=50, workers=num_workers)
    model.save(model_name)
    print("Palavras mais similares a 'presidente':\n", model.most_similar("presidente"))




if __name__ == "__main__":
    pass
    #save_locally()
    # train_w2v_model()
    train_w2v_model_per_article()


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
import time
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.phrases import Phrases
import logging
from string import punctuation
import re
import string
import matplotlib
from matplotlib import pyplot as plt

from itertools import combinations
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn import metrics
from itertools import cycle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                    level=logging.INFO)

nltk.download('stopwords')
sw = stopwords.words('portuguese') + list(string.punctuation)

pattern = re.compile(r"[^-a-zA-Z\s]")

# Set values for various parameters
num_features = 300  # Word vector dimensionality
min_word_count = 40  # Minimum word count
num_workers = 8  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words


def get_phrases(doc):
    """
    Quebra o texto limpo em frases
    :param doc:
    :return:
    """
    frases = sent_tokenize(doc['cleaned_text'])
    frases = [wordpunct_tokenize(frase.lower().strip().strip(punctuation)) for frase in frases if frase.strip()]
    return frases


def sentence_gen(limit=20e6):
    """
    Carrega as frases da coleção local para treinamento do modelo do W2v
    :return:
    """
    con = pymongo.MongoClient('localhost', port=27017)
    col = con.word2vec.frases
    for doc in col.find({'frases': {'$exists': True}}, {'frases': 1, '_id': 0}, limit=limit):
        for frase in doc['frases']:
            if frase == []:
                continue
            frase = [w.strip(punctuation) for w in frase if w not in sw]
            if frase == []:
                continue
            # print(frase)
            # print(type(frase[0]))
            yield frase


def bigram_gen(limit=20e6):
    for sentence in sentence_gen(limit):
        yield bigram[sentence]


def trigram_gen(limit=20e6):
    for sentence in sentence_gen(limit):
        yield trigram[sentence]


def text_gen(limit=2e6):
    con = pymongo.MongoClient('localhost', port=27017)
    col = con.word2vec.frases
    for doc in col.find({'cleaned_text': {'$exists': True}}, {'cleaned_text': 1, '_id': 0}, limit=limit):
        text = pattern.sub("", doc['cleaned_text'])
        yield wordpunct_tokenize(text.lower())


def train_w2v_model(model_name="MediaCloud_w2v", n=50000, ngram=1):
    print("Training model...")
    t0 = time.time()
    if ngram == 2:
        gen = bigram_gen
        model_name += "_bigrams"
    elif ngram == 3:
        gen = trigram_gen
        model_name += "_trigrams"
    else:
        gen = sentence_gen
    model = Word2Vec(workers=num_workers, \
                     size=num_features, min_count=min_word_count, \
                     window=context, iter=1)  # an empty model, no training
    model.build_vocab(gen(n))  # can be a non-repeatable, 1-pass generator
    print("Levou {} segundos para construir o vocabulário".format(time.time() - t0))
    t0 = time.time()
    model.train(gen(n))  # can be a non-repeatable, 1-pass generator
    print("Levou {} segundos para treinar o modelo".format(time.time() - t0))

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    # model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model.save(model_name)
    print("Palavras mais similares a 'presidente':\n", model.most_similar("presidente"))


# def train_d2v_model(model_name="MediaCloud_d2v"):
#     model = Doc2Vec(sentences, size=100, window=8, min_count=45, workers=num_workers)
#     model.save(model_name)
#     print("Palavras mais similares a 'presidente':\n", model.most_similar("presidente"))


def train_w2v_model_per_article(model_name="MediaCloud_d2v", n=50000):
    t0 = time.time()
    model = Word2Vec(workers=num_workers, \
                     size=num_features, min_count=min_word_count, \
                     window=context, iter=3)  # an empty model, no training
    model.build_vocab(text_gen(n))  # can be a non-repeatable, 1-pass generator
    print("Levou {} segundos para construir o vocabulário".format(time.time() - t0))
    t0 = time.time()
    model.train(text_gen(n))  # can be a non-repeatable, 1-pass generator
    print("Levou {} segundos para treinar o modelo".format(time.time() - t0))
    model.save(model_name)
    print("Palavras mais similares a 'presidente':\n", model.most_similar("presidente"))





def cluster_vectors(model, nwords, method='DBS'):
    print("Calculating Clusters.")
    X = model.syn0[:nwords, :]
    if method == 'AP':
        af = AffinityPropagation(copy=False).fit(X)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        n_clusters_ = len(set(labels))
    elif method == 'DBS':
        print("Computing DBSCAN")
        db = DBSCAN(eps=0.03, min_samples=5, algorithm='brute', metric='cosine').fit(X)
        labels = db.labels_
        n_clusters_ = len(set(labels))
    elif method == 'AC':
        print("Computing Agglomerative Clustering")
        ac = AgglomerativeClustering(10).fit(X)
        labels = ac.labels_
        n_clusters_ = ac.n_clusters
    elif method == 'KM':
        print("Computing MiniBatchKmeans clustering")
        km = MiniBatchKMeans(n_clusters=300, batch_size=200).fit(X)
        labels = km.labels_
        n_clusters_ = len(km.cluster_centers_)

    print('Estimated number of clusters: %d' % n_clusters_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    # % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(labels_true, labels))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

    return X, labels


def extract_cluster(model, labels, label=1):
    """
    Extract the words associated with the cluster identified by the `label`

    :param model: Word2vec model
    :param labels: list with cluster labels attributed to each word in the model
    :param label: label if the cluster to extract
    :return:
    """
    indices = [i for i in range(len(labels)) if labels[i] == label]
    palavras = [model.index2word[i] for i in indices]
    return palavras


if __name__ == "__main__":
    pass
    ## Treina MOdelos
    # save_locally()
    print("Calculating Bigrams")
    # bigram = Phrases(sentence_gen(100000))
    print("Calculating Tigrams")
    # trigram = Phrases(bigram[sentence_gen(100000)])
    # train_w2v_model(n=1000000, ngram=2)
    # train_w2v_model(n=1000000, ngram=3)
    # train_w2v_model_per_article()

    ## Cluster analysis
    model = Word2Vec.load("MediaCloud_w2v")
    X, labels = cluster_vectors(model, 200000, 'KM')

    for i in range(20):
        print(extract_cluster(model, labels, i))

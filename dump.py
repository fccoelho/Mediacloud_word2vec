# -*- coding:utf-8 -*-
u"""
Created on 13/04/15
by fccoelho
license: GPL V3 or Later
"""
import pymongo
from string import punctuation
from nltk.tokenize import sent_tokenize, wordpunct_tokenize

def get_phrases(doc):
    frases = sent_tokenize(doc['cleaned_text'])
    frases = [wordpunct_tokenize(frase.lower().strip().strip(punctuation)) for frase in frases if frase.strip()]
    return frases

def querydb():
    con = pymongo.MongoClient('localhost', port=27000)
    db = con.MCDB
    col = db.articles
    for doc in col.find({"cleaned_text": {"$exists": True}}, {'cleaned_text': True, 'link': True, 'published': True}):
        yield doc

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

if __name__ == "__main__":
    pass
    # save_locally()

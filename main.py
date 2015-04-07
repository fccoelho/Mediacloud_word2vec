# -*- coding:utf-8 -*-
u"""
Created on 07/04/15
by fccoelho
license: GPL V3 or Later
"""

__docformat__ = 'restructuredtext en'


import pandas as pd
import pymongo
from nltk.tokenize import sent_tokenize

def querydb():
    con = pymongo.MongoClient('localhost', port=27000)
    db = con.MCDB
    col = db.articles
    for doc in col.find({"cleaned_text": {"$exists": True}}, {'cleaned_text': True, 'link': True, 'published': True}):
        yield doc

def get_phrases(doc):
    frases = sent_tokenize(doc['cleaned_text'])
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

if __name__ == "__main__":
    #save_locally()

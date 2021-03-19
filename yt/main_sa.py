#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:35:51 2021

@author: tatiana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
stopwords = stopwords.words('english')

data = pd.read_csv('../data/training.1600000.processed.noemoticon.csv',
                    encoding='latin-1',
                    header = None,
                    usecols=[0,5],
                    names=['target','text'])

#random sample of only 50K samples
data = data.sample(n=50000,random_state=100).reset_index(drop=True)

# delete stopwords (they say it does not need for real sentiment extraction)
def clean_text(text):
    """
    Return clean text
    params
    ------------
        text: string
    """

    text = text.lower() #lowercase
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if not t in stopwords] #remove stopwords
    tokens = [t for t in tokens if t.isalnum()] #remove punctuation
    text_clean = " ".join(tokens)
    
    return text_clean

#clean text 
data['text'] = [clean_text(text) for text in data['text']]


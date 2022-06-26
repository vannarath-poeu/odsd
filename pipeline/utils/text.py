import pandas as pd
import numpy as np
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def init():
  nltk.download('stopwords')
  nltk.download('punkt')

def remove_stopwords(text):
  stop_words = stopwords.words('english')
  more_stopwords = ['u', 'im', 'c']
  stop_words = stop_words + more_stopwords
  text = ' '.join(word for word in text.split(' ') if word not in stop_words)
  return text

def clean_text(text):
  '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
  and remove words containing numbers.'''
  text = str(text).lower()
  text = re.sub(r'\[.*?\]', '', text)
  text = re.sub(r'https?://\S+|www\.\S+', '', text)
  text = re.sub(r'<.*?>+', '', text)
  text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub(r'\n', ' ', text)
  text = re.sub(r'\w*\d\w*', '', text)
  return text

def stemm_text(text):
  stemmer = nltk.SnowballStemmer("english")
  text = ' '.join(stemmer.stem(word) for word in text.split(' '))
  return text
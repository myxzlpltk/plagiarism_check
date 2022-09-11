import re
import string
import nltk
import gensim.downloader as api
import numpy as np
import timeit
from itertools import groupby

from pymilvus import Collection
from pymilvus import connections
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from joblib import load
from statistics import mean

# Flask APP
app = Flask(__name__)

# Server Variables
print('Init server variables...')
scaler = load('models/scaler.joblib')
sentences_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
wl = WordNetLemmatizer()

print('Init MilvusDB...')
connections.connect(
    alias="default",
    host='localhost',
    port='19530'
)
collection = Collection('documents')
collection.load()

print('Loading model...')
start = timeit.default_timer()
wv = api.load('word2vec-google-news-300')
index2word_set = set(wv.index_to_key)
stop = timeit.default_timer()
print(f'Model loaded in {stop - start} seconds!')


@app.route("/")
def main():
    return render_template('input.html')


@app.route('/result', methods=['POST'])
def result():
    texts = request.form['texts']
    texts = sentences_splitter.tokenize(texts)

    clean_texts = [preprocess(text) for text in texts]

    vectors = [avg_feature_vector(text, model=wv, num_features=300, index=index2word_set) for text in clean_texts]
    vectors = scaler.transform(vectors)
    vectors = [vector / np.linalg.norm(vector) for vector in vectors]

    query_result = collection.search(
        data=vectors,
        anns_field='vector',
        param=dict(metric_type='IP', params=dict(nprobe=1)),
        limit=1,
        output_fields=['title', 'text', 'hash']
    )

    results = []
    reports = []
    score = 0
    for i, item in enumerate(query_result):
        if len(item) > 0 and item[0].distance > 0.85:
            results.append(item[0].id)
            reports.append({
                'index': i,
                'id': item[0].id,
                'title': item[0].entity.get('title'),
                'text': item[0].entity.get('text'),
                'hash': item[0].entity.get('hash'),
                'distance': round(item[0].distance * 100, 1)
            })
            score += 1 - item[0].distance
        else:
            score += 1
            results.append(None)

    reports = sorted(reports, key=lambda x: x['hash'])
    reports = [(key, list(group)) for key, group in groupby(reports, key=lambda x: x['hash'])]
    reports = sorted(reports, key=lambda x: mean([el['distance'] for el in x[1]]), reverse=True)
    score = 100 - round(score / len(vectors) * 100, 2)

    return render_template('output.html', data=zip(texts, results), reports=reports, score=score)


# Cleaning
def cleaning(s):
    # Lowercase text
    s = s.lower()
    # Trim text
    s = s.strip()
    # Remove punctuations, special characters, URLs & hashtags
    s = re.compile('<.*?>').sub('', s)
    s = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', s)
    s = re.sub('\s+', ' ', s)
    s = re.sub(r'\[[0-9]*\]', ' ', s)
    s = re.sub(r'[^\w\s]', '', str(s).lower().strip())
    s = re.sub(r'\d', ' ', s)
    s = re.sub(r'\s+', ' ', s)

    return s


# Remove stopword
def stopword(s):
    a = [i for i in s.split() if i not in stopwords.words('english')]
    return ' '.join(a)


# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Tokenize the sentence
def lemmatizer(s):
    word_pos_tags = nltk.pos_tag(word_tokenize(s))  # Get position tags
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in
         enumerate(word_pos_tags)]  # Map the position tag and lemmatize the word/token
    return " ".join(a)


# Preprocessing
def preprocess(s):
    s = cleaning(s)
    s = stopword(s)
    s = lemmatizer(s)
    return s


# Feature extraction
def avg_feature_vector(sentence, model, num_features, index):
    words = sentence.split()
    feature_vec = np.zeros((num_features,), dtype='float32')
    n_words = 0
    for word in words:
        if word in index:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

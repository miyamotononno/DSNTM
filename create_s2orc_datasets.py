import os
import time
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import collections
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
import re

cvectorizer = None
PICKLE_FILES = ['train.pickle', 'valid.pickle', 'test.pickle']
TXT_FILES = ['train.txt', 'valid.txt', 'test.txt']
PATH_EMBEDDING = "saved_emb.pickle"

def load_data(dir_dataset, file_name):
  with open(os.path.join(dir_dataset, file_name), 'rb') as f:
    data = pickle.load(f)
  return data

def dump_data(dir_dataset, files, values):
  for file, value in zip(files, values):
    with open(os.path.join(dir_dataset, file), 'wb') as f:
      pickle.dump(value, f)

def init_cvectorizer():
  def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text
  with open('stops.txt', 'r') as f:
    stops = f.read().split('\n')
  cvectorizer = CountVectorizer(min_df=10, stop_words=stops, max_df=0.7, preprocessor=preprocess_text)

def filter_dataset(times, doc_term, paper_ids):
  tokens = []
  counts = []
  time_data = []
  papers = []
  for i, (tim, doc, paper_id) in enumerate(zip(times, doc_term, paper_ids)):
    terms = doc.toarray()[0]
    tokens_by_doc = np.where(terms > 0)
    counts_by_doc = terms[tokens_by_doc]
    if len(counts_by_doc) == 0:
      print("no famous words: ", i)
      print(tim)
      continue

    tokens.append(tokens_by_doc[0])
    counts.append(counts_by_doc)
    time_data.append(tim)
    papers.append(paper_id)

  times_2years = [(tt-2006)//2 for tt in time_data]

  dataset = {
  'tokens': tokens,
  'counts': counts,
  'times': time_data,
  'paper_ids': papers,
  'doc_term': doc_term,
  'times_by_2years': times_2years
  }
  return dataset

def save_doc_dataset(df_dataset, dir_dataset, path_saved_embeddings, vocab=None):
  corpus = []
  times = []
  paper_ids = []
  df = df_dataset.query('2006 <= year < 2020').dropna(how='any')
  corpus = df['abstract'].values
  times = df['year'].astype('int32').values
  paper_ids = df['paper_id'].values

  # split 3:1:1 for train, valid and test
  corpus_train, corpus_rest, times_train, times_rest, paper_ids_train, paper_ids_rest \
    = train_test_split(corpus, times, paper_ids, train_size=0.6, random_state=1)
  corpus_valid, corpus_test, times_valid, times_test, paper_ids_valid, paper_ids_test \
    = train_test_split(corpus_rest, times_rest, paper_ids_rest, train_size=0.5, random_state=1)
  
  if vocab is not None:
    doc_term_train = cvectorizer.transform(corpus_train)
  else:
    doc_term_train = cvectorizer.fit_transform(corpus_train)
    vocab = cvectorizer.get_feature_names_out()
  doc_term_valid = cvectorizer.transform(corpus_valid)
  doc_term_test = cvectorizer.transform(corpus_test)
  
  print("[Size] doc_train: {}, doc_valid: {}, doc_test: {}".format(doc_term_train.shape[0], doc_term_valid.shape[0], doc_term_test.shape[0]))
  print("vocab size: ", len(vocab))
  with open(os.path.join(dir_dataset, 'vocab.pickle'), 'wb') as f:
    pickle.dump(vocab, f)

  save_embedding(vocab, out_path=path_saved_embeddings)

  data_train = filter_dataset(times_train, doc_term_train, paper_ids_train)
  data_valid = filter_dataset(times_valid, doc_term_valid, paper_ids_valid)
  data_test = filter_dataset(times_test, doc_term_test, paper_ids_test)
  dump_data(dir_dataset, PICKLE_FILES, [data_train, data_valid, data_test])
  return vocab

def save_coherence(dir_dataset):
  dir_coherence = os.path.join(dir_dataset, 'coherence')
  if not os.path.isdir(dir_coherence):
    os.makedirs(dir_coherence)
  for in_file, out_file in zip(PICKLE_FILES, TXT_FILES):
    dataset = load_data(dir_dataset, in_file)
    doc_term = dataset['doc_term']
    corpus = cvectorizer.inverse_transform(doc_term)
    with open(os.path.join(dir_coherence, out_file), 'wb') as f:
      for doc in corpus:
        f.write(" ".join(doc).encode())
        f.writelines(['\n'.encode()])

def save_citations(df_dataset, dir_dataset):
  dir_citation = os.path.join(dir_dataset, 'citation')
  if not os.path.isdir(dir_citation):
    os.makedirs(dir_citation)
  for pickle_file in PICKLE_FILES:
    dataset = load_data(dir_dataset, pickle_file)
    paper_ids = dataset['paper_ids']  
    papers = set()
    for pid in paper_ids:
      papers.add(int(pid))

    dict_citation = {}
    for i, raw in df_dataset.iterrows():
      cit = raw['outbound_citations']
      cit_ids = []
      for cid in eval(cit):
        c = int(cid)
        if c in papers:
          cit_ids.append(c)
      dict_citation[int(raw['paper_id'])] = cit_ids
    
    with open(os.path.join(dir_citation, pickle_file), 'wb') as f:
      pickle.dump(dict_citation, f)

def save_embedding(vocab, out_path, in_path='dataset/skipgram_emb_300d.txt'):
    vectors = {}
    with open(in_path, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if word in vocab:
                vect = np.array(line[1:]).astype(np.float64)
                vectors[word] = vect
    embeddings = np.zeros((len(vocab), len(vect)))
    words_found = 0
    for i, word in enumerate(vocab):
        try: 
            embeddings[i] = vectors[word]
            words_found += 1
        except KeyError:
            embeddings[i] = np.random.normal(scale=0.6, size=(len(vect), ))
    print("embeddings.shape: ", embeddings.shape)
    with open(out_path, 'wb') as f:
        pickle.dump(embeddings, f)

def create_datasets(df_dataset, dir_dataset, path_saved_embeddings):
  save_doc_dataset(df_dataset, dir_dataset, path_saved_embeddings)
  save_coherence(dir_dataset)
  save_citations(df_dataset, dir_dataset)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='acl', help='acl or cs')
  args = parser.parse_args()

  init_cvectorizer()
  if args.dataset == 'acl':
    dir_dataset = 'dataset/acl'
    df_dataset = pd.read_csv(os.path.join(dir_dataset, 'acl.csv'))
    path_saved_embeddings = os.path.join(dir_dataset, PATH_EMBEDDING)
  elif args.dataset == 'cs':
    dir_dataset = 'dataset/cs'
    df_dataset = pd.read_csv(os.path.join(dir_dataset, 'cs.csv'))
    path_saved_embeddings = os.path.join(dir_dataset, PATH_EMBEDDING)

  create_datasets(df_dataset, dir_dataset, path_saved_embeddings)
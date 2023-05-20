import os
import random
import pickle
import numpy as np
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_2d_list(expanded_tokens, file_path):
    f_path = os.path.join('dataset/coherence/time31', file_path)
    with open(f_path, 'wb') as f:
        for tokens in expanded_tokens:
            doc = " ".join(tokens)
            f.write(doc.encode())
            f.writelines(['\n'.encode()])

def expand_BoW(token_list, count_list, vocab):
    if len(token_list) != len(count_list):
        print("[error] must be same size between token_list and count_list")
        return

    corpus_list = [0] * len(token_list)
    for i, (tokens, counts) in enumerate(zip(token_list, count_list)):
        corpus = list()
        if len(tokens) > 1 or len(counts) > 1:
            print("Article_{} might have 2 or more docments. Check it", i)
        token = tokens[0]
        count = counts[0]
        for t, c in zip(token, count):
            corpus.extend([vocab[t]]*c)
        corpus_list[i] = corpus
    return corpus_list


def get_batch(tokens, counts, ind, vocab_size, emsize=300, temporal=False, times=None):
    """fetch input data by batch."""
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))
    if temporal:
        times_batch = np.zeros((batch_size, ))
    for i, doc_id in enumerate(ind):
        doc = tokens[doc_id]
        count = counts[doc_id]
        if temporal:
            timestamp = times[doc_id]
            times_batch[i] = timestamp
        elif len(doc) == 1: 
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
    data_batch = torch.from_numpy(data_batch).float().to(device)
    if temporal:
        times_batch = torch.from_numpy(times_batch).to(device)
        return data_batch, times_batch
    return data_batch

def get_rnn_input(tokens, counts, times, num_times, vocab_size):
    indices = torch.randperm(len(tokens))
    indices = torch.split(indices, 1000) 
    rnn_input = torch.zeros(num_times, vocab_size).to(device)
    cnt = torch.zeros(num_times, ).to(device)
    for idx, ind in enumerate(indices):
        data_batch, times_batch = get_batch(tokens, counts, ind, vocab_size, temporal=True, times=times)
        for t in range(num_times):
            tmp = (times_batch == t).nonzero()
            docs = data_batch[tmp].squeeze().sum(0)
            rnn_input[t] += docs
            cnt[t] += len(tmp)
    rnn_input = rnn_input / cnt.unsqueeze(1)
    return rnn_input

def load_vocab(path):
    with open(os.path.join(path, 'vocab.pickle'), 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def create_s2orc_dataset(path, vocab_size, mode):
    if mode == 'predict':
        mode = 'test'
    file_name = '{}.pickle'.format(mode)

    with open(os.path.join(path, file_name), 'rb') as f:
        data = pickle.load(f)

    rnn_inp = get_rnn_input(data['tokens'], data['counts'], data['times_by_2years'], len(np.unique(data['times_by_2years'])), vocab_size)
    return data['tokens'], data['counts'], data['times_by_2years'], rnn_inp, data['paper_ids']

def get_citation_matrix(target_year, citation_year, dict_time_to_papers, data_citations):
    target_paper_ids = dict_time_to_papers[target_year] 
    citation_paper_ids = dict_time_to_papers[citation_year] # 引用先
    citation_matrix = torch.zeros((len(target_paper_ids), len(citation_paper_ids)), device=device)
    cnt = 0
    for i, tpid in enumerate(target_paper_ids):
        target_citation_ids = data_citations[tpid]
        for cid in target_citation_ids:
            if cid in citation_paper_ids:
                citation_matrix[i][citation_paper_ids.index(cid)] = 1
                cnt += 1
    return citation_matrix

def create_bow_by_time(tokens, counts, times, paper_ids, args):
    dict_time_to_idx = {}
    for i, (t, pid) in enumerate(zip(times, paper_ids)):
        if t in dict_time_to_idx:
            dict_time_to_idx[t].append(i)
        else:
            dict_time_to_idx[t] = []
            dict_time_to_idx[t].append(i)
    
    dict_bow = {}
    for t in range(args.num_times):
        bow, _ = get_batch(tokens, counts, dict_time_to_idx[t], args.vocab_size, temporal=True, times=times)
        dict_bow[t] = bow

    return dict_bow

def init_citations(times, paper_ids, args, mode):
    dict_time_to_paper_id = {}
    for i, (t, pid) in enumerate(zip(times, paper_ids)):
        if t in dict_time_to_paper_id:
            dict_time_to_paper_id[t].append(pid)
        else:
            dict_time_to_paper_id[t] = []
            dict_time_to_paper_id[t].append(pid)

    dir_citation = os.path.join(args.data_path, 'citation')
    file_name = '{}.pickle'.format(mode)
    with open(os.path.join(dir_citation, file_name), 'rb') as f:
        citations = pickle.load(f)

    # dict_citation[t][t_cit] indicates the citation matrix which show if each paper at t_cit cites paper at t or not. 　 
    dict_citation = {}
    for t in range(args.num_times):
        citation_by_time = {}
        if t == 0:
            continue
        for t_cit in range(t):
            citation_matrix = get_citation_matrix(t, t_cit, dict_time_to_paper_id, citations)
            citation_by_time[t_cit] = citation_matrix
        dict_citation[t] = citation_by_time

    return dict_citation

def divide_dataset_by_time(tokens, counts, times, paper_ids, vocab_size):
    tokens_by_time = {}
    counts_by_time = {}
    paper_ids_by_time = {}
    for tok, cnt, tim, paper_id in zip(tokens, counts, times, paper_ids):
        if not tim in tokens_by_time:
            tokens_by_time[tim] = []
            counts_by_time[tim] = []
            paper_ids_by_time[tim] = []

        tokens_by_time[tim].append(tok)
        counts_by_time[tim].append(cnt)
        paper_ids_by_time[tim].append(paper_id)

    num_doc_by_time = {}
    for k, v in tokens_by_time.items():
        num_doc_by_time[k] = len(v)

    return tokens_by_time, counts_by_time, paper_ids_by_time, num_doc_by_time

def load_citation_file(args, mode='train'):
    if args.mode == 'train':
        file_path = 'citation/train.pickle'
    elif args.mode == 'valid':
        file_path = 'citation/valid.pickle'
    elif args.mode == 'test':
        file_path = 'citation/test.pickle' 
    else:
        file_path = 'citation/test.pickle'
    with open(os.path.join(args.data_path, file_path), 'rb') as f:
        dict_citations = pickle.load(f)
    return dict_citations

def load_original(args):
    if args.dataset == 'acl':
        file_path = 'acl.csv'
    else:
        file_path = 'cs.csv'
    df = pd.read_csv(os.path.join(args.data_path, file_path))
    return df

def get_embedding(args, vocab):
    emb_path = os.path.join(args.data_path, 'saved_emb.pickle')

    vectors = {}
    with open(args.emb_path, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if word in vocab:
                vect = np.array(line[1:]).astype(np.float64)
                vectors[word] = vect
    embeddings = np.zeros((len(vocab), args.emb_size))
    words_found = 0
    for i, word in enumerate(vocab):
        try: 
            embeddings[i] = vectors[word]
            words_found += 1
        except KeyError:
            embeddings[i] = np.random.normal(scale=0.6, size=(args.emb_size, ))
    with open(emb_path, 'wb') as f:
        pickle.dump(embeddings, f)
    embeddings = torch.from_numpy(embeddings).to(device)
    return embeddings

def prepare_model(args, embeddings, vocab, model_class):
    tokens, counts, times, rnn_inp, paper_ids \
        = create_s2orc_dataset(args.data_path, args.vocab_size, args.mode)
    dict_bow = create_bow_by_time(tokens, counts, times, paper_ids, args)
    dict_citation = None
    if args.citation:
        dict_citation = init_citations(times, paper_ids, args, args.mode)
    if args.wo_attention:
        model = model_class(args, embeddings, tokens, counts, times, rnn_inp, vocab)
    else:
        model = model_class(args, embeddings, tokens, counts, times, rnn_inp, vocab, dict_bow, dict_citation)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device) 
    if args.citation:
        model.update_citation_data(dict_citation, dict_bow)
    data = {
        'tokens': tokens,
        'counts': counts,
        'times': times,
        'paper_ids': paper_ids,
    }
    return model, data

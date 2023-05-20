#/usr/bin/python

import os
if not 'CUDA_VISIBLE_DEVICES' in os.environ:
    print("not specify cuda id")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import copy
import torch
import numpy as np 
import tracemalloc
import time
import data 

from torch import nn, optim
from torch.nn import functional as F

from dsntm import DSNTM, EvaluationModel
from dsntm_wo_attention import DSNTMWithoutAttention, EvaluationModelWithoutAttention
from utils import save_pickle_file, load_pickle_file, get_parameter_size


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_train(epoch, model, optimizer, train_tokens, train_counts, train_times, train_rnn_inp, train_dict_citation, train_dict_bow, args):
    """Train DSNTM on data for one epoch.
    """
    model.train()
    model.update_citation_data(train_dict_citation, train_dict_bow)
    acc_loss, acc_nll, acc_kl_theta_loss, acc_kl_eta_loss, acc_kl_alpha_loss = 0,0,0,0,0
    cnt = 0
    indices = torch.randperm(args.num_docs_train)
    indices = torch.split(indices, args.batch_size)
    for idx, ind in enumerate(indices):
        optimizer.zero_grad()
        model.zero_grad()
        data_batch, times_batch = data.get_batch(
            train_tokens, train_counts, ind, args.vocab_size, args.emb_size, temporal=True, times=train_times)
        sums = data_batch.sum(1).unsqueeze(1)
        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch

        loss, nll, kl_eta, kl_theta, kl_alpha, _ = model(data_batch, normalized_data_batch, times_batch, train_rnn_inp, args.num_docs_train)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        acc_loss += torch.sum(loss).item()
        acc_nll += torch.sum(nll).item()
        acc_kl_theta_loss += torch.sum(kl_theta).item()
        acc_kl_eta_loss += torch.sum(kl_eta).item()
        acc_kl_alpha_loss += torch.sum(kl_alpha).item()
        cnt += 1

    cur_loss = round(acc_loss / cnt, 2) 
    cur_nll = round(acc_nll / cnt, 2) 
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
    cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
    cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2)
    lr = optimizer.param_groups[0]['lr']
    print('Epoch----->{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch, lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_loss))
    return model

def main(args):
    ## get data
    # 1. vocabulary
    vocab = data.load_vocab(args.data_path)
    args.vocab_size = len(vocab)
    valid_tokens, valid_counts, valid_times, valid_rnn_inp, valid_paper_ids \
        = data.create_s2orc_dataset(args.data_path, args.vocab_size, 'valid')
    args.num_docs_valid = len(valid_tokens)
    args.num_times = len(np.unique(valid_times))
    valid_dict_bow = data.create_bow_by_time(valid_tokens, valid_counts, valid_times, valid_paper_ids, args)
    valid_dict_citation = None
    if args.citation:
        valid_dict_citation = data.init_citations(valid_times, valid_paper_ids, args, 'valid')

    if args.mode == 'train':
        print('\n')
        print('Training Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
        print('=*'*100)
    
    if not args.train_embeddings:
        embeddings = data.get_embedding(args, vocab)
    else:
        embeddings = None

    if args.wo_attention:
        model = DSNTMWithoutAttention(args, embeddings)
        valid_model = EvaluationModelWithoutAttention(args, embeddings, valid_tokens, valid_counts, valid_times, valid_rnn_inp, vocab)
    else:
        model = DSNTM(args, embeddings)
        valid_model = EvaluationModel(args, embeddings, valid_tokens, valid_counts, valid_times, valid_rnn_inp, vocab, valid_dict_bow, valid_dict_citation)
    
    if args.mode == 'train':
        print('\nDSNTM architecture: {}'.format(model))
    
    model.to(device)
    valid_model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    best_ppl, epoch_best_ppl = 1e9, 0
    best_model = None
    if args.mode == 'train' or args.mode == 'train_valid':
        train_tokens, train_counts, train_times, train_rnn_inp, train_paper_ids \
            = data.create_s2orc_dataset(args.data_path, args.vocab_size, 'train')
        args.num_docs_train = len(train_tokens)
        train_dict_bow = data.create_bow_by_time(train_tokens, train_counts, train_times, train_paper_ids, args)
        train_dict_citation = None
        if args.citation:
            train_dict_citation = data.init_citations(train_times, train_paper_ids, args, 'train')
        all_val_ppls = []
        lr_param_size, non_lr_param_size = get_parameter_size(model)
        print("Total Size of the learnable and non-learnable Parameter: ", lr_param_size , non_lr_param_size)
        torch.cuda.synchronize()
        start = time.time()
        tracemalloc.start()
        for epoch in range(1, args.epochs+1):
            model = run_train(epoch, model, optimizer, train_tokens, train_counts, train_times, train_rnn_inp, train_dict_citation, train_dict_bow, args)
            model.update_citation_data(valid_dict_citation, valid_dict_bow)
            valid_model.load_state_dict(model.state_dict())
            val_ppl = valid_model.get_perplexity()
            
            print('VAL PPL: {}'.format(val_ppl))
            print('*'*100)
            if best_ppl > val_ppl:
                best_ppl = val_ppl
                epoch_best_ppl = epoch
                if args.best_model == 'ppl':
                    best_model = copy.deepcopy(model)
            if args.best_model == 'last':
                best_model = model
            ## check whether to anneal lr
            lr = optimizer.param_groups[0]['lr']
            if args.anneal_lr and (len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
                print("decrease learning rate")
                optimizer.param_groups[0]['lr'] /= args.lr_factor
            all_val_ppls.append(val_ppl)
        
        torch.cuda.synchronize()
        elapsed_time = time.time() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        print(int(elapsed_time//60), 'min.')

        print("Best PPL: {} at Epoch {}".format(best_ppl, epoch_best_ppl))
        model = best_model
        model.eval()
        if len(args.model_path) > 0:
            model.update_citation_data(None, None) # reduce memory
            torch.save(model.state_dict(), args.model_path)

    cohrence_path = os.path.join(args.data_path, args.coherence_path)

    if args.mode == 'valid' or args.mode == 'train_valid':
        if args.mode == 'valid':
            valid_model.load_state_dict(torch.load(args.model_path))
        ppl, coherence, diversity = valid_model.get_performance(visualize=True, cohrence_path=cohrence_path)
        print("Perplexity: {}, Average Coherence: {}, Average Diversity: {}".format(ppl, round(coherence, 3), round(diversity, 3)))
        
    if args.mode == 'test':
        if args.wo_attention:
            test_model, test_dataset = data.prepare_model(args, embeddings, vocab, EvaluationModelWithoutAttention)
        else:
            test_model, test_dataset = data.prepare_model(args, embeddings, vocab, EvaluationModel)
        ppl, coherence, diversity = test_model.get_performance(visualize=False, cohrence_path=cohrence_path)
        print("Perplexity: {}, Average Coherence: {}, Average Diversity: {}".format(ppl, round(coherence, 3), round(diversity, 3)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The Dynamic Structured Neural Topic Model')

    ### data and file related arguments
    parser.add_argument('--dataset', type=str, default='s2orc_acl', help='name of corpus')
    parser.add_argument('--data_path', type=str, default='dataset/acl', help='directory containing data')
    parser.add_argument('--emb_path', type=str, default='dataset/skipgram_emb_300d.txt', help='file path containing embeddings')
    parser.add_argument('--coherence_path', type=str, default='coherence', help='directory path containing data for coherence')
    parser.add_argument('--model_path', default= 'file path to save model results', type=str)
    parser.add_argument('--batch_size', type=int, default=512, help='number of documents in a batch for training')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='number of documents in a batch for validation')
    parser.add_argument('--best_model', type=str, default='ppl', help='last or ppl')

    ### model-related arguments
    parser.add_argument('--wo_attention', action='store_true', help='remove self-attention from DSNTM')
    parser.add_argument('--citation', action='store_true', help='use citation data and include citation loss')
    parser.add_argument('--c_weight', type=float, default=1.0, help='the weight of citation loss')
    parser.add_argument('--train_embeddings', type=int, default=0, help='whether to fix rho or train it')
    parser.add_argument('--num_topics', type=int, default=20, help='number of topics')
    parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
    parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
    parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
    parser.add_argument('--eta_nlayers', type=int, default=3, help='number of layers for eta')
    parser.add_argument('--eta_hidden_size', type=int, default=200, help='number of hidden units for rnn')
    parser.add_argument('--delta', type=float, default=0.005, help='prior variance')

    ### optimization-related arguments
    parser.add_argument('--lr', type=float, default=0.0006, help='learning rate')
    parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--mode', type=str, default='train', help='train, valid or test')
    parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
    parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
    parser.add_argument('--eta_dropout', type=float, default=0.0, help='dropout rate on rnn for eta')
    parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
    parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
    parser.add_argument('--anneal_lr', type=int, default=1, help='whether to anneal the learning rate or not')
    parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

    args = parser.parse_args()

    if torch.cuda.is_available():
        print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
        torch.cuda.empty_cache()

    ## set seed
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    main(args)

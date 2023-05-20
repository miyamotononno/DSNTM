import torch
import pickle
import data

def save_pickle_file(values, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(values, f)

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        values = pickle.load(f)
    return values

def check_topic_integrating(args, target_paper_ids, source_paper_ids):
    """
    Args:
    args: info
    target_paper_ids: torch.tensor. 1d-array
    source_paper_ids: torch.tensor. 2d-array

    """
    df = data.load_original(args)
    dict_citations = data.load_citation_file(args)
    for pid in target_paper_ids:
        pid = pid.item()
    
        for c_pid in dict_citations[pid]:
            source_ids = torch.nonzero(source_paper_ids==c_pid)
            if len(source_ids) > 0:
                print("="*20)
                print("Citing Paper: ")
                print(df[df['paper_id']==pid]['title'].values[0])
                print()
                print("Cited Paper Year : ",  int(df[df['paper_id']==c_pid]['year'].values[0]))
                print(df[df['paper_id']==c_pid]['title'].values[0])
                print(source_ids)
                print("="*20)
                print()

def check_topic_branching(args, target_paper_ids, source_paper_ids, citing_paper_info):
    df = data.load_original(args)
    dict_citations = data.load_citation_file(args)

    for paper_idx, pid in enumerate(torch.flatten(source_paper_ids)):
        pid = int(pid.item())
        if not pid in dict_citations:
            continue

        for c_pid in dict_citations[pid]:
            target_ids = torch.nonzero(target_paper_ids==c_pid)
            if len(target_ids) > 0:
                print("="*20)
                print("Cited Paper Year : ",  int(df[df['paper_id']==c_pid]['year'].values[0]))
                print(df[df['paper_id']==c_pid]['title'].values[0])
                print()
                print("Citing Paper info: ", citing_paper_info[paper_idx])
                print("Citing Paper Year : ",  int(df[df['paper_id']==pid]['year'].values[0]))
                print(df[df['paper_id']==pid]['title'].values[0])
                
def get_parameter_size(net):
    lr_params = 0
    non_lr_params = 0
    for p in net.parameters():
        if p.requires_grad:
            lr_params += p.numel()
        else:
            non_lr_params += p.numel()

    return lr_params, non_lr_params
    
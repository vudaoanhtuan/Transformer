import os
import torch
import torch.nn as nn

from .Transformer import Transformer


def get_model(src_vocab, trg_vocab, d_model=512, d_ff=2048, n_layers=6, heads=8, dropout=0.1, device=-1):
    assert d_model % heads == 0
    assert dropout < 1

    model = Transformer(src_vocab, trg_vocab, d_model, d_ff, n_layers, heads, dropout)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) 
    
    if device != -1:
        model = model.cuda()
    
    return model

def load_model(model, optim=None, sched=None, path=''):
    if os.path.isfile(path):
        state = torch.load(path)
        model.load_state_dict(state['model'])
        if optim is not None:
            optim.load_state_dict(state['optim'])
        if sched is not None:
            sched.load_state_dict(state['sched'])
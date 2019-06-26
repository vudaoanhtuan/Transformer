import time
import datetime
import os
import sys
import re
import logging

import torch
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset, Multi30k
import spacy

from Model.utils import *
from Model.Mask import *
from Model.Optim import *
from load_data import *


logging.basicConfig(filename='train.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

HOME_PATH = './'
WEIGHT_PATH = os.path.join(HOME_PATH, "weight")

if not os.path.exists(WEIGHT_PATH):
    os.makedirs(WEIGHT_PATH)


def forward_and_loss(model, src, trg, loss_fn, src_pad=1, trg_pad=1):
    trg_input = None
    if trg is not None:
        trg_input = trg[:, :-1]
       
    src_mask, trg_mask = create_mask(src, trg_input, src_pad_token=src_pad, trg_pad_token=trg_pad)
    preds = model(src, trg_input, src_mask, trg_mask)
    ys = trg[:, 1:].contiguous().view(-1)

    loss = loss_fn(preds.view(-1, preds.size(-1)), ys, ignore_index=trg_pad)
    return preds, loss
    

def train_model(model, optimizer, train_iter, src_pad=1, trg_pad=1, scheduler=None, save_path=None):
    total_loss = 0.0
    total_item = 0

    model.train()
    
    for i, batch in enumerate(train_iter): 

        src = batch.src.transpose(0,1)
        trg = batch.trg.transpose(0,1)

        optimizer.zero_grad()
        _, loss = forward_and_loss(model, src, trg, F.cross_entropy, src_pad=src_pad, trg_pad=trg_pad)
        
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        total_item += batch.batch_size
        
    # Save model
    if save_path is not None:
        state = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": None if scheduler is None else scheduler.state_dict(),
        }
        
        torch.save(state, os.path.join(WEIGHT_PATH, "cp_epoch_%d.h5" % (epoch_num)))


def  evaluate_model(model, val_iter, src_pad=1, trg_pad=1):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_item = 0
        for i, batch in enumerate(val_iter):
            src = batch.src.transpose(0,1)
            trg = batch.trg.transpose(0,1)

            _, loss = forward_and_loss(model, src, trg, F.cross_entropy, src_pad=src_pad, trg_pad=trg_pad)
            
            total_loss += loss.item()
            total_item += batch.batch_size
        


EN, DE = load_field('multi30k.h5')

train, val, test = Multi30k.splits(('.en', '.de'), (EN, DE), root='./data')

DEVICE = torch.device('cuda') if torch.cuda.is_available() else -1
BATCH_SIZE = 32

train_iter = BucketIterator(
    train,
    batch_size=BATCH_SIZE,
    device=DEVICE
)

val_iter = BucketIterator(
    val,
    batch_size=BATCH_SIZE,
    device=DEVICE
)

SCR_NUMWORD = len(EN.vocab.itos)
TRG_NUMWORD = len(DE.vocab.itos)
LR = 0.0001

print("Init model")
model = get_model(SCR_NUMWORD,TRG_NUMWORD, device=DEVICE) # base model
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
sched = CosineWithRestarts(optimizer, T_max=len(train_iter))


CURRENT_EPOCH = 0
NUM_EPOCH = 30

print("Load model")
load_model(model, optim=optimizer, sched=sched, path=os.path.join(WEIGHT_PATH, 'cp_epoch_%d.h5' % CURRENT_EPOCH))


print("Training")
for e in range(CURRENT_EPOCH+1, CURRENT_EPOCH+1+NUM_EPOCH):
    train_model(model, optimizer, train_iter, val_iter, epoch_num=e, scheduler=sched)
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

from tqdm import tqdm

from Model.utils import *
from Model.Mask import *
from Model.Optim import *


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
    
    with tqdm(total=len(train_iter)) as pbar:
        for batch in train_iter: 

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

            pbar.update(1)
            pbar.set_description("loss = %.6f" % (total_loss/total_item))
            
    # Save model
    if save_path is not None:
        state = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": None if scheduler is None else scheduler.state_dict(),
        }
        
        torch.save(state, os.path.join(WEIGHT_PATH, "cp_epoch_%d.h5" % (epoch_num)))


def evaluate_model(model, val_iter, src_pad=1, trg_pad=1):
    model.eval()
    with torch.no_grad(), tqdm(total=len(val_iter)) as pbar:
        total_loss = 0.0
        total_item = 0
        for batch in val_iter:
            src = batch.src.transpose(0,1)
            trg = batch.trg.transpose(0,1)

            _, loss = forward_and_loss(model, src, trg, F.cross_entropy, src_pad=src_pad, trg_pad=trg_pad)
            
            total_loss += loss.item()
            total_item += batch.batch_size

            pbar.update(1)
            pbar.set_description("val_loss = %.6f" % (total_loss/total_item))




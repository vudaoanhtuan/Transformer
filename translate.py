import sys
import math
import re

import spacy
import torch

from load_data import *
from Model.utils import get_model, load_model


from beamsearch import translate_sentence



if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (SRC, TRG), (train_ds, val_ds, test_ds) = load_multi30k()

    SCR_NUMWORD = len(SRC.vocab.itos)
    TRG_NUMWORD = len(TRG.vocab.itos)

    model = get_model(SCR_NUMWORD, TRG_NUMWORD)

    if DEVICE.type == 'cuda':
        model = model.cuda()

    load_model(model, path="weight/epoch_18.h5")


    with open(".data/multi30k/test2016.en", encoding='utf8') as f:
        test = f.read().split('\n')[:-1]

    with open(".data/multi30k/tran2016.de", 'w', encoding='utf8', ) as f:
        for i, s in enumerate(test):
            t = translate_sentence(s, model, SRC, TRG, device=DEVICE)
            f.write(t)
            f.write('\n')
            sys.stdout.write('\r')
            sys.stdout.write("\tTranslating... [%.2f%%]" % ( (i+1)*100.0/len(test)))
            sys.stdout.flush()
        



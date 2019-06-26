import re

import spacy
import torch
from torchtext.data import Field, Dataset
from torchtext.datasets import Multi30k

class tokenize(object):
    
    def __init__(self, lang):
        self.nlp = spacy.load(lang)
            
    def tokenizer(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]


def load_multi30k():
    t_src = tokenize('en')
    t_trg = tokenize('de')

    MAX_LEN = 20

    EN = Field(lower=True, tokenize=t_src.tokenizer, fix_length=MAX_LEN)
    DE = Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>', fix_length=MAX_LEN)

    train, val, test = Multi30k.splits(('.en', '.de'), (EN, DE))

    MAX_NUMWORD = 37000

    EN.build_vocab(train, max_size=MAX_NUMWORD)
    DE.build_vocab(train, max_size=MAX_NUMWORD)

    return (EN, DE), (train, val, test)


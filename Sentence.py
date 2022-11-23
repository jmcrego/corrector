import re
#import sys
#import copy
#import time
#import json
import random
import logging
#import argparse
#import pyonmttok
#import numpy as np
#from Misspell import Misspell
#from Replacement import Replacement
#from Spurious import Spurious
#from collections import defaultdict

JOINER = '￭'
PUNC = ',.;:!?\'"«»<>'
IS_WORD = re.compile(r'^[A-Za-zÀ-ÿ]+$')
IS_PUNC = re.compile(r'^'+JOINER+'?['+PUNC+']'+JOINER+'?$') #one punctuation char as in PUNC with or without preceding/succeding joiners 
NONE = 'NONE' # used for error_type or word_to_predict
#SEPAR = 'ǁ'

class Word():
    def __init__(self, txt, error_type, word_to_predict, is_noisy):
        assert len(txt) == len(error_type) == len(word_to_predict), 'different number of tokens: txt={} error_type={}, word_to_predict={}'.format(txt,error_type,word_to_predict)
        self.txt = txt                         ### list with string forms (original or noised)
        self.error_type = error_type           ### list of error_types referred to self.txt (must contain as many elements as self.txt)
        self.word_to_predict = word_to_predict ### list containing the word_to_predict (must contain as many elements as self.txt)
        self.is_noisy = is_noisy               ### True or False

    def is_word(self):
        return len(self.txt)==1 and IS_WORD.match(self.txt[0]) is not None

    def is_punc(self):
        return len(self.txt)==1 and IS_PUNC.match(self.txt[0]) is not None
        
    def __call__(self, triplet=False):
        if not triplet:
            return ' '.join(self.txt)
        out = []
        for i in range(len(self.txt)):
            t = [self.txt[i]]
            if self.error_type[i] != NONE:
                t.append(self.error_type[i])
                if self.word_to_predict[i] != NONE:
                    t.append(self.word_to_predict[i])
            else:
                assert self.error_type[i] == self.word_to_predict[i] , 'if error_type={} is NONE word_to_predict={} SHOULD be NONE'.format(self.error_type[i],self.word_to_predict[i])
            out.append(t)
        return out


class Sentence():
    def __init__(self, toks):
        self.words = [Word([t],[NONE],[NONE],False) for t in toks]

    def __call__(self, triplet=False):
        if not triplet:
            return ' '.join([w() for w in self.words]).split()
        out = []
        for w in self.words:
            out += w(triplet)
        return out
    
    def __len__(self):
        return len(self.words)

    def __getitem__(self, i):
        return self.words[i]

    def get_random_unnoised_idx(self):
        idxs = [idx for idx in range(len(self.words)) if not self.words[idx].is_noisy]
        if len(idxs) == 0:
            return None
        random.shuffle(idxs)
        return idxs[0]
    

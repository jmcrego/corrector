#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging
from collections import defaultdict

class Vocab():
    def __init__(self, f):
        self.vocab = defaultdict()
        self.VOCAB = []
        #
        self.PAD = '<PAD>'
        self.idx_PAD = 0
        self.vocab[self.PAD] = len(self.vocab)
        self.VOCAB.append(self.PAD)
        #
        self.UNK = '<UNK>'
        self.idx_UNK = 1
        self.vocab[self.UNK] = len(self.vocab)
        self.VOCAB.append(self.UNK)
        #
        with open(f,'r') as fd:
            for n,l in enumerate(fd):
                wrd = l.rstrip()
                if wrd in self.vocab:
                    logging.info('Repeated entry {} in lines {} {} vocab'.format(wrd,n+1,self.vocab[wrd]-1))
                    continue
                self.vocab[wrd] = len(self.vocab)
                self.VOCAB.append(wrd)
        logging.info('Loaded Vocab {} ({} entries)'.format(f,len(self.VOCAB)))
        
    def __contains__(self, s):
        return s in self.vocab
        
    def __getitem__(self, s):
        ### return a string
        if type(s) == int:
            return self.VOCAB[s]
        ### return an index
        if s in self.vocab:
            return self.vocab[s]
        return self.idx_UNK

    def __len__(self):
        return len(self.vocab)

    
if __name__ == '__main__':
                    
    l = Vocab('resources/french.dic.50k')

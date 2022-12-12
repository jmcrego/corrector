import re
import sys
import time
import random
import logging
import argparse
import pyonmttok
import unicodedata
from transformers import AutoTokenizer, T5Tokenizer, T5TokenizerFast, MT5TokenizerFast, MT5Tokenizer

JOINER = '￭'
SPACER = '▁'
PUNC = ',.;:!?\'"«»<>'
IS_WORD = re.compile(r'^[A-Za-zÀ-ÿ]+$')
IS_PUNC = re.compile(r'^['+PUNC+']$') #one punctuation char as in PUNC
#IS_PUNC = re.compile(r'^'+JOINER+'?['+PUNC+']'+JOINER+'?$') #one punctuation char as in PUNC with or without preceding/succeding joiners

class Error():
    def __init__(self, t=None, i=None, e=None):
        self.t = t #string to generate as correction (without joiners) or None. Ex: 'auprès'
        self.i = i #[ids] of the word to generate or None. Ex: [2, 257]
        self.e = e #error tag. Ex: 'SUB:I'

    def __call__(self, txt=False):
        if txt:
            return self.t
        return {'t': self.t, 'i': self.i, 'e': self.e}
        
class Word():
    def __init__(self, t=None, i=None, e=None): #t='￭-￭' i=[3, 18]
        assert t != JOINER, 'Word: token {} cannot be a single JOINER'.format(t)  
        self.starts_with_joiner = True if t.startswith(JOINER) else False
        self.ends_with_joiner = True if t.endswith(JOINER) else False
        t = t.replace(JOINER, '')
        self.t = t #string form (Ex: 'auprès')
        self.i = i #ids of this token (Ex: [2, 257])
        self.e = e #may be None (when word is not noisy) or an instance of Error
        
    def __call__(self, txt=False):
        if txt:
            return (JOINER if self.starts_with_joiner else '') + self.t + (JOINER if self.ends_with_joiner else '')
        return {'t': self.t, 'i': self.i, '<':self.starts_with_joiner, '>':self.ends_with_joiner, 'e': self.e() if isinstance(self.e, Error) else self.e}

    def is_word(self):
        if self.t is None:
            return None
        return IS_WORD.match(self.t) is not None

    def is_punc(self):
        if self.t is None:
            return None
        return IS_PUNC.match(self.t) is not None

class Sentence():
    def __init__(self, tok, lids): #['C', "￭'￭", 'est', 'mon', 'premier', 'exemple', '￭.'], [[205], [3, 31], [259], [1911], [2761], [5300], [3, 5]]
        assert len(tok) == len(lids)
        self.words = [ Word(tok[i],lids[i]) for i in range(len(tok)) ]

    def __call__(self, txt=False):
        return [w(txt) for w in self.words]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, i):
        return self.words[i]

    def get_random_unnoised_idx(self):
        idxs = [idx for idx in range(len(self.words)) if self.words[idx].e is None]
        if len(idxs) == 0:
            return None
        random.shuffle(idxs)
        return idxs[0]
    

class Tokenizer():
    def __init__(self, path="t5-base", nofast=False):
        self.max_onmttok_len = 300
        self.onmt = pyonmttok.Tokenizer('aggressive', joiner_annotate=True)
        if path == 't5-base':
            self.subtok = T5Tokenizer.from_pretrained(path, model_max_length=512) if nofast else T5TokenizerFast.from_pretrained(path, model_max_length=512)
            self.remove_initial = 0
            self.remove_ending = -1 ### corresponding to </s>
        elif path == 'google/mt5-base':
            self.subtok = MT5Tokenizer.from_pretrained(path, model_max_length=512) if nofast else MT5TokenizerFast.from_pretrained(path, model_max_length=512)
            self.remove_initial = 0
            self.remove_ending = -1 ### corresponding to </s>
        elif path == 'facebook/nllb-200-distilled-600M':
            self.subtok = AutoTokenizer.from_pretrained(path, src_lang="fra_Latn", model_max_length=512)
            self.remove_initial = 0
            self.remove_ending = -2 ### corresponding to </s>, fra_Latn
        else:
            logging.error('unrecognised path={}'.format(path))
            sys.exit()
            
    def tok(self, l): #C'est mon premier exemple.
        t = self.onmt(l) #['C', "￭'￭", 'est', 'mon', 'premier', 'exemple', '￭.']
        if len(t) > self.max_onmttok_len:
            t = t[:self.max_onmttok_len]
        #logging.debug('Tokenizer:t:\t{}'.format(t))
        return t

    def detok(self, t):
        return self.onmt.detokenize(t)
    
    def sub(self, l): #C ￭'￭ est mon premier exemple ￭.
        ### subtokenization must be applied over tokens already tokenized otherwise the mapping cannot be computed between subtokens and tokens (words)
        i = self.subtok(l)["input_ids"][self.remove_initial:self.remove_ending] #[205, 3, 31, 259, 1911, 2761, 5300, 3, 5]
        t2 = self.subtok.convert_ids_to_tokens(i) #['▁C', '▁', "'", '▁est', '▁mon', '▁premier', '▁exemple', '▁', '.'] 
        #logging.debug('Tokenizer:i:\t{}'.format(i))
        #logging.debug('Tokenizer:t2:\t{}'.format(t2))
        return i, t2
    
    def sub2tok(self, t2, i):
        assert len(t2) == len(i)
        i2 = []
        for n,t in enumerate(t2):
            if t.startswith(SPACER):
                i2.append([])
            i2[-1].append(i[n])
        #logging.debug('Tokenizer:i2:\t{}'.format(i2))
        return i2 #[[205], [3, 31], [259], [1911], [2761], [5300], [3, 5]]

    def __call__(self, l, only_tok=False, p_prefix=0.0, p_lcfirst=0.0, spurious=None):
        t = self.tok(l) #['C', "￭'￭", 'est', 'mon', 'premier', 'exemple', '￭.']
        t = self.prefix_lcfirst(t,p_prefix,p_lcfirst,spurious)
        if only_tok:
            return t
        ### map subtokens (t2,i2) to tokens (t,i)
        l = ' '.join(t).replace(JOINER,'') #C ' est mon premier exemple .
        i, t2 = self.sub(l) #[205, 3, 31, 259, 1911, 2761, 5300, 3, 5] ['▁C', '▁', "'", '▁est', '▁mon', '▁premier', '▁exemple', '▁', '.']
        i2 = self.sub2tok(t2,i) #[[205], [3, 31], [259], [1911], [2761], [5300], [3, 5]]
        if len(t) != len(i2):
            return [], []
        return t, i2
    
    def prefix_lcfirst(self, t, p_prefix, p_lcfirst, spurious):
        self.is_prefix = False
        self.is_lcfirst = False
        if p_prefix == 0.0 and p_lcfirst == 0.0:
            return t
        ### take a prefix of the entire sentence ###
        if len(t)>1 and random.random() < p_prefix: 
            last = random.randint(1,min(9,len(t)-1)) #both included
            t = t[:last]
            self.is_prefix = True
        ### lowercase first word ###################
        if spurious is not None and t[0][0] != t[0][0].lower() and t[0][0].lower()+t[0][1:] in spurious and random.random() < p_lcfirst:
            t[0] = t[0][0].lower()+t[0][1:]
            self.is_lcfirst = True
        return t
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='t5-base', type=str, help="huggingface tokenizer path (t5-base)")
    parser.add_argument("--nofast", action='store_true', help="do not use tokenizer fast")
    parser.add_argument("--only_tok", action='store_true', help="only tokenize")
    parser.add_argument("--debug", action='store_true', help="logging level=DEBUG (INFO)")
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'DEBUG' if args.debug else 'INFO'), filename=None, filemode = 'w')
    
    token = Tokenizer(args.path, args.nofast)
    tic = time.time()
    for n,l in enumerate(sys.stdin):
        res = token(l.rstrip(), only_tok=args.only_tok)
        if args.only_tok:
            print(token.detok(res))
        else:
            print(Sentence(res[0],res[1])())
    toc = time.time()
    logging.info('Parsed {} lines in {:.2f} sec ({:.2f} lines/sec)'.format(n+1,toc-tic,(n+1)/(toc-tic)))

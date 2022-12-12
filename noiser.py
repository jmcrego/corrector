import re
import sys
import copy
import time
import json
import random
import logging
import argparse
import pyonmttok
import numpy as np
from collections import defaultdict
from HFTokenizer import JOINER
from Misspell import Misspell
from Spurious import Spurious
from Replacement import Replacement
from HFTokenizer import HFTokenizer, Sentence, Word, PUNC, Error

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

class Noiser():
    def __init__(self, cfg):
        self.cfg = cfg
        self.M = Misspell(Dict2Class(cfg.misspell), 'SUB:M')
        self.G = Replacement(cfg.inflect, 'SUB:I')
        self.H = Replacement(cfg.homophone, 'SUB:H')
        self.S = Spurious(cfg.spurious)
        self.HFTokenizer = HFTokenizer(cfg.path, nofast=False)
        self.noises_seen = {n:1 for n in self.cfg.noises.keys()} ### initially all noises appeared once
        self.n_noises2weights = {}
        self.total_sents_with_n_noises = [1] * (self.cfg.max_n+1)
        self.total_sents_with_n_noises[0] += 10 #to avoid selecting 0 the first 10 sentences
        self.total_sents_with_noises = 0
        self.total_sentences = 0
        self.total_noises = 0
        self.total_tokens = 0
        self.total_prefixed = 0
        self.total_lcfirst = 0

    def stats(self, tictoc):
        logging.info('{:.1f} seconds'.format(tictoc))
        logging.info('{} sentences'.format(self.total_sentences))
        logging.info('{} tokens'.format(self.total_tokens))
        logging.info('{:.1f} sentences/sec'.format(self.total_sentences/(tictoc)))
        logging.info('{:.1f} tokens/sec'.format(self.total_tokens/(tictoc)))
        logging.info('{:.1f} tokens/sentence'.format(self.total_tokens/self.total_sentences))
        logging.info('{:.1f} noises/sentence'.format(self.total_noises/self.total_sentences))
        logging.info('{} ({:.2f}%) noisy sentences'.format(self.total_sents_with_noises,100.0*self.total_sents_with_noises/self.total_sentences))
        logging.info('{} ({:.2f}%) prefixed sentences'.format(self.total_prefixed,100.0*self.total_prefixed/self.total_sentences))
        logging.info('{} ({:.2f}%) lcfirst sentences'.format(self.total_lcfirst,100.0*self.total_lcfirst/self.total_sentences))
        if self.total_tokens:
            logging.info('{} ({:.2f}%) noisy tokens'.format(self.total_noises,100.0*self.total_noises/self.total_tokens))
        if self.total_noises:
            logging.info('*** Noises ***')
            for k, v in sorted(self.noises_seen.items(),key=lambda item: item[1], reverse=True):
                if v>1:
                    logging.info("{}\t{:.2f}%\t1-every-{:.1f}\t{}".format(v-1,100.0*(v-1)/self.total_noises,self.total_tokens/(v-1),k))
            logging.info('*** Misspells ***')
            self.M.stats()
            logging.info('*** Sentences with n-noises ***')
            self.total_sents_with_n_noises[0] -= 10 ### to counterbalance the addition in constructor
            for k, v in enumerate(self.total_sents_with_n_noises):
                if v>1:
                    logging.info("{}\t{:.2f}%\t1-every-{:.1f}\t{}-noises".format(v-1,100.0*(v-1)/self.total_sents_with_noises,self.total_sentences/(v-1),k))
        
    def __call__(self, l):
        self.t_ini, lids = self.HFTokenizer(l,p_prefix=self.cfg.p_prefix,p_lcfirst=self.cfg.p_lcfirst,spurious=self.S) #['C', "￭'￭", 'est', 'mon', 'premier', 'exemple', '￭.'] [[205], [3, 31], [259], [1911], [2761], [5300], [3, 5]]
        self.t_ini_detok = self.HFTokenizer.detok(self.t_ini) ### self.t_ini_detok may be a prefix/lcfirst/trunc of l
        self.s = Sentence(self.t_ini,lids)
        self.total_sentences += 1
        self.total_tokens += len(self.s)
        self.total_prefixed += 1 if self.HFTokenizer.is_prefix else 0
        self.total_lcfirst += 1 if self.HFTokenizer.is_lcfirst else 0
        add_n_noises = self.add_n_noises()
        curr_n_noises = 0
        #logging.debug('trying to inject {} noises'.format(add_n_noises))
        for _ in range(add_n_noises): ### try to add n noises in this sentence
            idx = self.s.get_random_unnoised_idx() #get an unnoised token
            #logging.debug('trying to noise idx {} {}'.format(idx,self.s.words[idx]()))
            if idx == None:
                break
            for next_noise, _ in self.sort_noises(): ### try all noises sorted by frequency (lower first)
                #logging.debug('trying to inject noise {}'.format(next_noise))
                if next_noise == 'inflect':
                    res = self.replace(idx,self.G)
                elif next_noise == 'homophone':
                    res = self.replace(idx,self.H)
                elif next_noise == 'misspell':
                    res = self.misspell(idx)
                elif next_noise == 'case':
                    res = self.case(idx)
                elif next_noise == 'copy':
                    res = self.copy(idx)
                elif next_noise == 'swap':
                    res = self.swap(idx)
                elif next_noise == 'space_add':
                    res = self.space_add(idx)
                elif next_noise == 'space_del':
                    res = self.space_del(idx)
                elif next_noise == 'hyphen_add':
                    res = self.hyphen_add(idx)
                elif next_noise == 'hyphen_del':
                    res = self.hyphen_del(idx)
                elif next_noise == 'punctuation':
                    res = self.punctuation(idx)
                elif next_noise == 'spurious_add':
                    res = self.spurious_add(idx,self.S)
                elif next_noise == 'spurious_del':
                    res = self.spurious_del(idx,self.S)
                else:
                    res = False
                if res:
                    self.noises_seen[next_noise] += 1
                    curr_n_noises += 1
                    break
        #logging.debug('added {} out of {} noises'.format(curr_n_noises,add_n_noises))
        self.total_sents_with_n_noises[curr_n_noises] += 1
        self.total_noises += curr_n_noises
        self.total_sents_with_noises += 1 if curr_n_noises else 0
        self.t_end = self.s(txt=True)
        self.t_end_detok = self.HFTokenizer.detok(self.t_end)
        ### some words in self.s (those noised) does not have ids. ids must be computed for the entire (noised) sentence
        i, t2 = self.HFTokenizer.sub(' '.join(self.t_end).replace(JOINER,''))
        i2 = self.HFTokenizer.sub2tok(t2,i)
        assert len(i2) == len(self.t_end)
        for k in range(len(self.s)):
            if self.s[k].i is None:
                self.s[k].i = i2[k] #logging.debug('assigning i2={} to token t2={} corresponding to {}'.format(i2[k], self.t_end[k], self.s[k]()))
        #logging.debug('t_ini: {}'.format(self.HFTokenizer.onmt.detokenize(self.t_ini)))
        #logging.debug('t_end: {}'.format(self.HFTokenizer.onmt.detokenize(self.t_end)))
        return self.s()

    def sort_noises(self):
        next_noises = {}
        for noise,tokens_per_noise in self.cfg.noises.items():
            next_noises[noise] = 1.0 * self.cfg.noises[noise] / self.noises_seen[noise]
        sorted_noises = sorted(next_noises.items(), key=lambda item:item[1], reverse=True)
        return sorted_noises#[0:5]
        
    def add_n_noises(self):
        max_n = min(int(len(self.s)*self.cfg.max_r), self.cfg.max_n)
        W = self.total_sents_with_n_noises[0:max_n+1] #weight of n noises is (inversely) proportional to its frequency
        add_n = np.argsort(W)
        return add_n[0]

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    def filter_if(self, idx, has_error=False, not_word=False, not_punc=False, not_word_or_punc=False, starts_with_joiner=False, ends_with_joiner=False, not_hyphen=False, min_len=0):
        if idx < 0 or idx >= len(self.s.words):
            return True
        if min_len and len(self.s.words[idx].t) < min_len:
            return True
        if has_error and self.s.words[idx].e is not None:
            return True
        if not_word and not self.s.words[idx].is_word():
            return True
        if not_punc and not self.s.words[idx].is_punc():
            return True
        if not_hyphen and not self.s.words[idx].t == '-':
            return True
        if not_word_or_punc and not self.s.words[idx].is_punc() and not self.s.words[idx].is_word():
            return True
        if starts_with_joiner and idx>=0 and idx<len(self.s.words) and self.s.words[idx].starts_with_joiner:
            return True
        if ends_with_joiner and idx>=0 and idx<len(self.s.words) and self.s.words[idx].ends_with_joiner:
            return True
        return False
    
    def misspell(self, idx): #deux
        if self.filter_if(idx, has_error=True, not_word=True):
            return False
        w = self.s.words[idx]
        txt, err = self.M(w.t)
        if txt == '' or err == '':
            return False
        self.s.words[idx] = Word(t=txt, e=Error(t=w.t, i=w.i, e=err)) #deuc|[int]|SUB:M
        logging.debug('NOISE misspell\t{}'.format(self.s.words[idx]())) 
        return True

    def replace(self, idx, R): #avais
        if self.filter_if(idx, has_error=True, not_word=True):
            return False
        w = self.s.words[idx]
        txt, err = R(w.t)
        if txt == '' or err == '':
            return False
        self.s.words[idx] = Word(t=txt, e=Error(t=w.t, i=w.i, e=err)) #avait|[int]|SUB:H
        logging.debug('NOISE replace\t{}'.format(self.s.words[idx]()))
        return True

    def punctuation(self, idx): #,
        if self.filter_if(idx, has_error=True, not_punc=True):
            return False
        err = 'SUB:P'
        w = self.s.words[idx]
        p = w.t[0]
        other_p = [x for x in PUNC if x!=p]
        random.shuffle(other_p)
        txt = (JOINER if w.starts_with_joiner else '') + other_p[0] + (JOINER if w.ends_with_joiner else '')
        self.s.words[idx] = Word(t=txt, e=Error(t=w.t, i=w.i, e=err)) #￭.|[int]|SUB:P
        logging.debug('NOISE punctuation\t{}'.format(self.s.words[idx]()))
        return True
    
    def copy(self, idx): #dans
        if self.filter_if(idx, has_error=True, not_word_or_punc=True) or self.filter_if(idx-1, ends_with_joiner=True) or self.filter_if(idx+1,starts_with_joiner=True):
            return False
        err = 'DEL'
        w = self.s.words[idx]
        txt = (JOINER if w.starts_with_joiner else '') + w.t + (JOINER if w.ends_with_joiner else '')
        self.s.words[idx].e = Error(t=None, i=None, e=None)
        self.s.words.insert(idx+1, Word(txt, i=w.i, e=Error(t=None, i=None, e=err))) #None|None|DEL
        logging.debug('NOISE copy\t{}'.format(self.s.words[idx+1]()))
        return True 

    def case(self, idx):
        if self.filter_if(idx, has_error=True, not_word=True):
            return False
        w = self.s.words[idx]
        if w.t.isupper(): ################################################## IBM
            err = 'CASE:X'
            if len(w.t) > 1 and random.random() < 0.5:
                txt = w.t[0]+w.t[1:].lower()   #Ibm
            else:
                txt = w.t.lower()              #ibm
        elif w.t.islower(): ################################################ ibm
            err = 'CASE:x'
            if len(w.t) > 1 and random.random() < 0.5:
                txt = w.t[0].upper()+w.t[1:]   #Ibm
            else:
                txt = w.t.upper()              #IBM
        elif len(w.t)>1 and w.t[0].isupper() and w.t[1:].islower(): #### Ibm
            err = 'CASE:Xx'
            if random.random() < 0.5:
                txt = w.t.upper()              #IBM
            else:
                txt = w.t.lower()              #ibm
        else: ################################################################ IbM
            return False
        self.s.words[idx] = Word(t=txt, e=Error(t=None, i=None, e=err)) #None|None|CASE:x
        logging.debug('NOISE case\t{}'.format(self.s.words[idx]()))
        return True
    
    def space_add(self, idx): #gestion
        minlen = 2
        if self.filter_if(idx, has_error=True, not_word=True, min_len=minlen*2):
            return False
        err = 'JOIN'
        w = self.s.words[idx]
        w_t = w.t
        k = random.randint(minlen,len(w_t)-minlen)
        wprev = w_t[:k]
        wpost = w_t[k:]
        self.s.words.pop(idx)
        self.s.words.insert(idx, Word(t=wpost, e=Error(t=None, i=None, e=None))) #ion  None|None|None
        self.s.words.insert(idx, Word(t=wprev, e=Error(t=None, i=None, e=err))) #gest  None|None|JOIN
        logging.debug('NOISE space_add\t{} {}'.format(self.s.words[idx](), self.s.words[idx+1]()))
        return True

    def space_del(self, idx): #mon ami
        if self.filter_if(idx, has_error=True, not_word=True) or self.filter_if(idx+1, has_error=True, not_word=True):
            return False
        err = 'DIV'
        w1, w2 = self.s.words[idx], self.s.words[idx+1]
        w1_t = w1.t
        w1_i = w1.i
        w2_t = w2.t
        self.s.words.pop(idx) #w1 deleted
        self.s.words[idx] = Word(t=w1_t+w2_t, e=Error(t=w1_t, i=w1_i, e=err)) ### monami|DIV|mon
        logging.debug('NOISE space_del\t{}'.format(self.s.words[idx]()))
        return True 

    def hyphen_del(self, idx): #anti - douleur
        if self.filter_if(idx, has_error=True, not_word=True) or self.filter_if(idx+1, has_error=True, not_hyphen=True) or self.filter_if(idx+2, has_error=True, not_word=True):
            return False
        err = 'ADD'
        w1, w2, w3 = self.s.words[idx], self.s.words[idx+1], self.s.words[idx+2]
        w2_t = w2.t #-
        w2_i = w2.i 
        self.s.words[idx].e = Error(t=w2_t, i=w2_i, e=err) ### anti   -|[int]|ADD
        self.s.words.pop(idx+1) #w2 is deleted
        self.s.words[idx+1].e = Error(t=None, i=None, e=None) ### douleur   None|None|None
        logging.debug('NOISE hyphen_del\t{}\t{}'.format(self.s.words[idx](),self.s.words[idx+1]()))
        return True

    def hyphen_add(self, idx): #grands magasins
        minlen = 4
        if self.filter_if(idx, has_error=True, not_word=True, min_len=minlen) or self.filter_if(idx+1, has_error=True, not_word=True, min_len=minlen):
            return False
        err = 'DEL'
        self.s.words[idx].e = Error(t=None, i=None, e=None)
        self.s.words[idx+1].e = Error(t=None, i=None, e=None)
        self.s.words.insert(idx+1, Word(t=JOINER+'-'+JOINER,  e=Error(t=None, i=None, e=err)))
        logging.debug('NOISE hyphen_add\t{}\t{}\t{}'.format(self.s.words[idx](),self.s.words[idx+1](),self.s.words[idx+2]()))
        return True

    def swap(self, idx):
        if self.filter_if(idx, has_error=True, not_word=True) or self.filter_if(idx+1, has_error=True, not_word=True) or self.filter_if(idx-1,ends_with_joiner=True) or self.filter_if(idx+2,starts_with_joiner=True):
            #cannot swap first/last tokens
            return False
        err = 'SWAP'
        if idx >= len(self.s.words)-1:
            return False
        w1, w2 = self.s.words[idx], self.s.words[idx+1]
        w1_t = w1.t
        w1_i = w1.i
        w2_t = w2.t
        w2_i = w2.i
        self.s.words[idx]   = Word(t=w2_t, i=w2_i, e=Error(t=None, i=None, e=err))
        self.s.words[idx+1] = Word(t=w1_t, i=w1_i, e=Error(t=None, i=None, e=None))
        logging.debug('NOISE swap\t{}\t{}'.format(self.s.words[idx](), self.s.words[idx+1]()))
        return True

    def spurious_add(self, idx, S):
        if self.filter_if(idx, has_error=True, not_word=True) or self.filter_if(idx+1, starts_with_joiner=True):
            #cannot add after last token
            return False
        err = 'DEL'
        w = self.s.words[idx]
        txt = S() #get an spurious word
        self.s.words.insert(idx+1,Word(t=txt, e=Error(t=None, i=None, e=err)))
        logging.debug('NOISE spurious_add\t{}'.format(self.s.words[idx+1]()))
        return True
        
    def spurious_del(self, idx, S):
        if self.filter_if(idx, has_error=True, not_word=True) or self.filter_if(idx+1, has_error=True, not_word=True) or self.filter_if(idx-1, ends_with_joiner=True) or self.filter_if(idx+1, starts_with_joiner=True) :
            return False
        err = 'ADD'
        w2 = self.s.words[idx+1]
        w2_t = w2.t
        w2_i = w2.i
        if w2_t not in S: #only delete spurious words
            return False
        self.s.words.pop(idx+1) #w2 is deleted
        self.s.words[idx].e = Error(t=w2_t, i=w2_i, e=err) ### w2|[int]|ADD
        logging.debug('NOISE spurious_del\t{}'.format(self.s.words[idx]()))
        return True
    
###################################################################################################
### MAIN ##########################################################################################
###################################################################################################

if __name__ == '__main__':
    example = {"inflect": "resources/Morphalou3.1_CSV.csv.inflect", "homophone": "resources/Morphalou3.1_CSV.csv.homophone", "spurious": "resources/Morphalou3.1_CSV.csv.spurious", "misspell": {"delete": 1, "repeat": 1, "close": 1, "swap": 1, "diacritics": 4, "consd": 25, "phone": 100}, "noises": {"inflect": 50, "homophone": 2, "punctuation": 2, "hyphen_add": 1, "hyphen_del": 100, "misspell": 1, "case": 1, "space_add": 1, "space_del": 1, "copy": 1, "swap": 1, "spurious_add": 1, "spurious_del": 1}, "max_r": 0.5, "max_n": 10, "seed": 23}
    parser = argparse.ArgumentParser(description="Tool to noise clean text following a noiser configuration file. Example: {}".format(example))
    parser.add_argument('config', type=str, help='noiser configuration file')
    parser.add_argument('--seed', type=int, default=0, help='seed for randomness (0)')
    parser.add_argument("--debug", action='store_true', help="logging level=DEBUG (INFO)")
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO' if not args.debug else 'DEBUG', None), filename=None, filemode = 'w')
    with open(args.config,'r') as fd:
        config = json.load(fd)
    if args.seed != 0:
        config['seed'] = args.seed
    random.seed(config['seed'])

    tic = time.time()
    n = Noiser(Dict2Class(config))
    for l in sys.stdin:
        noisy_tok = n(l.rstrip())
        print("{}\t{}\t{}".format(n.t_ini_detok, n.t_end_detok, noisy_tok))
    toc = time.time()
    n.stats(toc-tic)

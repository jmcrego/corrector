import re
import sys
import copy
import time
import json
import random
import argparse
import logging
import pyonmttok
from tqdm import tqdm
from Misspell import Misspell
from Replacement import Replacement
from collections import defaultdict

JOINER = '￭'
SEPAR = 'ǁ'
PUNCTS = ',.;:!?\'"«»<>'
WORD = re.compile(r'^[A-Za-zÀ-ÿ]+$')
PUNC = re.compile(r'^￭?[,.;:!?\'"«»<>]￭?$')
KEEP = 'KEEP'

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

class Word():
    def __init__(self, txt):
        self.ini = txt  ### original txt
        self.fin = None ### final txt (after noise) or None
        self.err = None ### error type or None
        self.is_word = WORD.match(self.ini) is not None 
        self.is_punc = PUNC.match(self.ini) is not None
        
    def set(self, ini=None, fin=None, err=None):
        if ini is not None:
            self.ini = ini
        if fin is not None:
            self.fin = fin
        if err is not None:
            self.err = err
        logging.debug('{} => {} => {}'.format(self.ini,self.fin,self.err))
            
    def __call__(self, final=True):
        return self.ini if not final or self.fin is None else self.fin

    def final_and_noise(self):
        if self.err is None or self.fin is None:
            return self.ini
        return ' '.join([i+(SEPAR+j if j!=KEEP else '') for i, j in zip(self.fin.split(), self.err.split())])
        
    def is_noised(self):
        return self.err is not None 
    
class Sentence():
    def __init__(self, toks):
        self.words = [Word(t) for t in toks]

    def __call__(self, final=True):
        return ' '.join([w(final) for w in self.words]).split()

    def final_and_noise(self):
        return ' '.join([w.final_and_noise() for w in self.words])
        
    def __len__(self):
        return len(self.words)

    def __getitem__(self, i):
        return self.words[i]

    def n_noises(self):
        return sum([w.is_noised() for w in self.words])

    def noises(self):
        return [w.err.replace(' '+KEEP,'').replace(KEEP+' ','') for w in self.words if w.err is not None and w.err != KEEP]
    
    def get_random_unnoised_idx(self):
        idxs = [idx for idx in range(len(self.words)) if not self.words[idx].is_noised()]
        if len(idxs) == 0:
            return None
        random.shuffle(idxs)
        return idxs[0]
    
class Noiser():
    def __init__(self, cfg):
        self.cfg = cfg
        self.M = Misspell(Dict2Class(cfg.misspell))
        self.G = Replacement(cfg.grammar, 'grammar')
        self.H = Replacement(cfg.homophone, 'homophone')
        self.tokenizer = pyonmttok.Tokenizer("aggressive", joiner_annotate=True, joiner=JOINER)
        self.n_tokens = 0
        self.errors_seen = {'grammar':1, 'homophone':1, 'misspell':1, 'case':1, 'duplicate':1, 'swap':1, 'space_add':1, 'space_del':1, 'hyphen_add':1, 'hyphen_del':1, 'punctuation':1}

    def sort_noises(self):
        next_errors = {}
        for error,tokens_per_error in self.cfg.noises.items():
            if error in self.errors_seen and tokens_per_error > self.n_tokens / self.errors_seen[error]:
                continue
            next_errors[error] = 1.0 / (self.errors_seen[error] * tokens_per_error)
        return sorted(next_errors.items(), key=lambda item: item[1], reverse=True)
        
    def __call__(self, l):
        l = l.rstrip()
        self.s = Sentence(self.tokenizer(l))
        self.n_tokens += len(self.s)
        add_n_noises = random.randint(int(len(self.s)*self.cfg.min_r), min(self.cfg.max_n,int(len(self.s)*self.cfg.max_r))) #number of noises to inject in current sentence, [min_noises, max_noises]
        n_attempts = 0
        while self.s.n_noises() < add_n_noises and n_attempts < 2*add_n_noises:
            n_attempts += 1
            idx = self.s.get_random_unnoised_idx()
            if idx == None:
                break
            #for next_noise in sorted(self.cfg.noises.keys(), key=lambda k: random.random()): #self.sort_noises():
            for next_noise, _ in self.sort_noises():
                res = False
                if next_noise == 'grammar': # and       self.errors_seen['grammar']/self.n_tokens < 1/self.cfg.noises['grammar']:
                    res = self.replace(idx,self.G)
                elif next_noise == 'homophone': # and   self.errors_seen['homophone']/self.n_tokens < 1/self.cfg.noises['homophone']:
                    res = self.replace(idx,self.H)
                elif next_noise == 'misspell': # and    self.errors_seen['misspell']/self.n_tokens < 1/self.cfg.noises['misspell']:
                    res = self.misspell(idx)
                elif next_noise == 'case': # and        self.errors_seen['case']/self.n_tokens < 1/self.cfg.noises['case']:
                    res = self.case(idx)
                elif next_noise == 'duplicate': # and   self.errors_seen['duplicate']/self.n_tokens < 1/self.cfg.noises['duplicate']:
                    res = self.duplicate(idx)
                elif next_noise == 'swap': # and        self.errors_seen['swap']/self.n_tokens < 1/self.cfg.noises['swap']:
                    res = self.swap(idx)
                elif next_noise == 'space_add': # and   self.errors_seen['space_add']/self.n_tokens < 1/self.cfg.noises['space_add']:
                    res = self.space_add(idx)
                elif next_noise == 'space_del': # and   self.errors_seen['space_del']/self.n_tokens < 1/self.cfg.noises['space_del']:
                    res = self.space_del(idx)
                elif next_noise == 'hyphen_add': # and  self.errors_seen['hyphen_add']/self.n_tokens < 1/self.cfg.noises['hyphen_add']:
                    res = self.hyphen_add(idx)
                elif next_noise == 'hyphen_del': # and  self.errors_seen['hyphen_del']/self.n_tokens < 1/self.cfg.noises['hyphen_del']:
                    res = self.hyphen_del(idx)
                elif next_noise == 'punctuation': # and self.errors_seen['punctuation']/self.n_tokens < 1/self.cfg.noises['punctuation']:
                    res = self.punctuation(idx)
                if res:
                    self.errors_seen[next_noise] += 1
                    break

        return self.tokenizer.detokenize(self.s()), self.s.final_and_noise(), self.s.noises()

    def misspell(self, idx):
        if idx < 0 or idx >= len(self.s.words):
            return False
        w = self.s.words[idx]
        if w.is_noised() or not w.is_word: # or w.has_joiner() or w.is_number():
            return False
        txt, err = self.M(w.ini)
        if txt == '' or err == '':
            return False
        w.set(fin=txt,err=err)
        return True

    def replace(self, idx, R):
        if idx < 0 or idx >= len(self.s.words):
            return False
        w = self.s.words[idx]
        if w.is_noised() or not w.is_word: # or w.has_joiner() or w.is_number():
            return False
        txt, err = R(w.ini)
        if txt == '' or err == '':
            return False
        w.set(fin=txt,err=err)
        return True

    def punctuation(self, idx): #may have joiner
        if idx < 0 or idx >= len(self.s.words):
            return False
        w = self.s.words[idx]
        if w.is_noised() or not w.is_punc: #one single punctuation with (or without) joiners
            return False
        p = w.ini[1] if w.ini[0] == JOINER else w.ini[0]
        other_p = [x for x in PUNCTS if x!=p]
        random.shuffle(other_p)
        fin = (JOINER if w.ini[0] == JOINER else '') + other_p[0] + (JOINER if w.ini[-1] == JOINER else '')
        w.set(fin=fin, err='punctuation')
        return True
    
    def duplicate(self, idx):
        if idx < 0 or idx >= len(self.s.words):
            return False
        w = self.s.words[idx]
        if w.is_noised() or not w.is_word: # or w.has_joiner() or w.is_number():
            return False
        w.set(fin=w.ini+' '+w.ini, err=KEEP+' '+'duplicate')
        return True 

    def case(self, idx):
        if idx < 0 or idx >= len(self.s.words):
            return False
        w = self.s.words[idx]
        if w.is_noised() or not w.is_word: # or w.has_joiner() or w.is_number():
            return False
        if w.ini.isupper(): #### IBM
            if len(w.ini) > 1 and random.random() < 0.5:
                w.set(fin=w.ini[0]+w.ini[1:].lower(), err='case:X') #Ibm
                return True
            else:
                w.set(fin=w.ini.lower(), err='case:X')              #ibm
                return True
        elif w.ini.islower(): #### ibm
            if len(w.ini) > 1 and random.random() < 0.5:
                w.set(fin=w.ini[0].upper()+w.ini[1:], err='case:x') #Ibm
                return True
            else:
                w.set(fin=w.ini.upper(), err='case:x')              #IBM
                return True
        elif len(w.ini)>1 and w.ini[0].isupper() and w.ini[1:].islower(): #### Ibm
            if random.random() < 0.5:
                w.set(fin=w.ini.upper(), err='case:Xx')             #IBM
                return True
            else:
                w.set(fin=w.ini.lower(), err='case:Xx')             #ibm
                return True
        return False
    
    def space_add(self, idx): #gestion
        if idx < 0 or idx >= len(self.s.words):
            return False
        w = self.s.words[idx]
        if w.is_noised() or not w.is_word: # or w.has_joiner() or w.is_number():
            return False
        minlen = 3
        if len(w.ini) < 2*minlen: #minimum length of resulting spltted tokens
            return False
        k = random.randint(minlen,len(w.ini)-minlen)
        prev = w.ini[:k]
        post = w.ini[k:]
        w.set(fin=prev+' '+post, err='space:del'+' '+KEEP) ### gest ion
        return True

    def space_del(self, idx): #mon ami
        if idx < 0 or idx >= len(self.s.words)-1:
            return False
        w1, w2 = self.s.words[idx], self.s.words[idx+1]
        if w1.is_noised() or not w1.is_word: # or w1.has_joiner() or not w1.is_word() or w1.is_number():
            return False
        if w2.is_noised() or not w2.is_word: #or w2.has_joiner() or not w2.is_word() or w2.is_number():
            return False
        w1.set(ini=w1.ini+' '+w2.ini, fin=w1.ini+w2.ini, err='space:add') ### monami
        self.s.words.pop(idx+1) #w2 must be deleted from list of words
        return True 

    def hyphen_del(self, idx):
        if idx < 0 or idx >= len(self.s.words)-2:
            return False
        w1, w2, w3 = self.s.words[idx], self.s.words[idx+1], self.s.words[idx+2]
        if w1.is_noised() or not w1.is_word: #or w1.has_joiner() or not w1.is_word() or w1.is_number():
            return False
        if w2.is_noised() or w2.ini != JOINER+'-'+JOINER:
            return False
        if w3.is_noised() or not w3.is_word : #or w3.has_joiner() or not w3.is_word() or w3.is_number():
            return False
        if random.random() < 0.5:
            w1.set(ini=w1.ini+' '+w2.ini+' '+w3.ini, fin=w1.ini+w3.ini, err='hyphen:add:split') ### antidouleur
        else:
            w1.set(ini=w1.ini+' '+w2.ini+' '+w3.ini, fin=w1.ini+' '+w3.ini, err='hyphen:add:merge'+' '+KEEP) ### anti douleur
        self.s.words.pop(idx+1) #w2 must be deleted from list of words
        self.s.words.pop(idx+1) #w3 must be deleted from list of words
        return True

    def hyphen_add(self, idx): #grands magasins
        min_words_len = 4
        if idx < 0 or idx >= len(self.s.words)-1:
            return False
        w1, w2 = self.s.words[idx], self.s.words[idx+1]
        if w1.is_noised() or not w1.is_word: #or w1.has_joiner() or not w1.is_word() or w1.is_number():
            return False
        if w2.is_noised() or not w2.is_word: #or w2.has_joiner() or not w2.is_word() or w2.is_number():
            return False
        if len(w1.ini) < min_words_len or len(w2.ini) < min_words_len:
            return False
        w1.set(ini=w1.ini+' '+w2.ini, fin=w1.ini+' '+JOINER+'-'+JOINER+' '+w2.ini, err=KEEP+' '+'hyphen:del'+' '+KEEP) ### grands #-# magasins
        self.s.words.pop(idx+1) #w2 must be deleted from list of words
        return True

    def swap(self, idx):
        if idx < 0 or idx >= len(self.s.words)-1:
            return False
        w1, w2 = self.s.words[idx], self.s.words[idx+1]
        if w1.is_noised() or not w1.is_word: #or w1.has_joiner() or not w1.is_word() or w1.is_number():
            return False
        if w2.is_noised() or not w2.is_word: #or w2.has_joiner() or not w2.is_word() or w2.is_number():
            return False
        w1.set(fin=w2.ini, err='swap:next')
        w2.set(fin=w1.ini, err=KEEP)
        return True

###################################################################################################
### MAIN ##########################################################################################
###################################################################################################

if __name__ == '__main__':
    example = {"grammar": "resources/Morphalou3.1_CSV.csv.grammar", "homophone": "resources/Morphalou3.1_CSV.csv.homophone", "misspell": {"delete": 1, "repeat": 1, "close": 1, "swap": 1, "diacritics": 10, "consd": 100, "phone": 100}, "noises": {"grammar": 1000, "homophone": 30, "punctuation": 1, "hyphen_add": 1, "hyphen_del": 1000, "misspell": 1, "case": 1, "space_add": 1, "space_del":1, "duplicate": 1, "swap": 1}, "min_r": 0.0, "max_r": 0.5, "max_n": 10, "seed": 23}
    parser = argparse.ArgumentParser(description="Tool to noise clean text following a noiser configuration file. Example: {}".format(example))
    parser.add_argument('config', type=str, default=0, help='noiser configuration file')
    parser.add_argument("--debug", action='store_true', help="logging level=DEBUG (INFO)")
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO' if not args.debug else 'DEBUG', None))
    with open(args.config,'r') as fd:
        config = json.load(fd)
    random.seed(config['seed'])

    tic = time.time()
    n = Noiser(Dict2Class(config))
    noises_seen = defaultdict(int)
    sents_with_n_noises = defaultdict(int)
    sents_with_noises = 0
    n_sentences = 0
    n_noises = 0
    n_tokens = 0
    #for l in sys.stdin:
    L = [l for l in sys.stdin]
    if not args.debug:
        pbar = tqdm(total=len(L))
    for l in L:
        noised, toknoised_and_noise, noises = n(l)
        print(noised+'\t'+toknoised_and_noise)
        n_sentences += 1
        n_tokens += len(toknoised_and_noise.split())
        n_noises += len(noises)
        sents_with_n_noises[len(noises)] += 1
        sents_with_noises += 1 if len(noises) else 0
        for e in noises:
            noises_seen[e] += 1
        if not args.debug:
            pbar.update(1)
            
    toc = time.time()
    logging.info('{:.1f} seconds, {} sentences ({} with noises), {} tokens ({} with noises)'.format(toc-tic,n_sentences,sents_with_noises,n_tokens,n_noises))
    logging.info('{:.1f} sentences/sec, {:.1f} tokens/sec'.format(n_sentences/(toc-tic), n_tokens/(toc-tic)))
    logging.info('{:.1f} tokens/sentence, {:.1f} noises/sentence'.format(n_tokens/n_sentences, n_noises/n_sentences))
    if n_noises:
        logging.info('Noises:')
        for k, v in sorted(noises_seen.items(),key=lambda item: item[1], reverse=True):
            logging.info("{}\t{:.2f}%\t{}".format(v,100.0*v/n_noises,k))
        logging.info('Sentences with n-noises:')
        for k, v in sorted(sents_with_n_noises.items(),key=lambda item: item[0], reverse=False):
            logging.info("{}\t{:.2f}%\t{}-noises".format(v,100.0*v/sents_with_noises,k))


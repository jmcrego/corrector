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
from Misspell import Misspell
from Replacement import Replacement
from collections import defaultdict
from Spurious import Spurious

JOINER = '￭'
SEPAR = 'ǁ'
PUNC = ',.;:!?\'"«»<>'
IS_WORD = re.compile(r'^[A-Za-zÀ-ÿ]+$')
IS_PUNC = re.compile(r'^'+JOINER+'?['+PUNC+']'+JOINER+'?$') #one punctuation char as in PUNC with or without preceding/succeding joiners 
NONE = 'NONE' # used for error_type or word_to_predict

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

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
    
class Noiser():
    def __init__(self, cfg):
        self.cfg = cfg
        self.M = Misspell(Dict2Class(cfg.misspell), 'SUB:M')
        self.G = Replacement(cfg.inflect, 'SUB:I')
        self.H = Replacement(cfg.homophone, 'SUB:H')
        self.S = Spurious(cfg.spurious)
        self.tokenizer = pyonmttok.Tokenizer("aggressive", joiner_annotate=True, joiner=JOINER)
        self.noises_seen = {n:1 for n in self.cfg.noises.keys()} ### initially all noises appeared once
        self.n_noises2weights = {}
        self.total_sents_with_n_noises = [1] * (self.cfg.max_n+1)
        self.total_sents_with_noises = 0
        self.total_sentences = 0
        self.total_noises = 0
        self.total_tokens = 0

    def stats(self, tictoc):
        logging.info('{:.1f} seconds'.format(tictoc))
        logging.info('{} sentences'.format(self.total_sentences))
        logging.info('{} tokens'.format(self.total_tokens))
        logging.info('{:.1f} sentences/sec'.format(self.total_sentences/(tictoc)))
        logging.info('{:.1f} tokens/sec'.format(self.total_tokens/(tictoc)))
        logging.info('{:.1f} tokens/sentence'.format(self.total_tokens/self.total_sentences))
        logging.info('{:.1f} noises/sentence'.format(self.total_noises/self.total_sentences))
        logging.info('{} ({:.2f}%) noisy sentences'.format(self.total_sents_with_noises,100.0*self.total_sents_with_noises/self.total_sentences))
        logging.info('{} ({:.2f}%) noisy tokens'.format(self.total_noises,100.0*self.total_noises/self.total_tokens))
        if self.total_noises:
            logging.info('*** Noises ***')
            for k, v in sorted(self.noises_seen.items(),key=lambda item: item[1], reverse=True):
                if v>1:
                    logging.info("{}\t{:.2f}%\t1-every-{:.1f}\t{}".format(v-1,100.0*(v-1)/self.total_noises,self.total_tokens/(v-1),k))
            logging.info('*** Misspells ***')
            self.M.stats()
            logging.info('*** Sentences with n-noises ***')
            for k, v in enumerate(self.total_sents_with_n_noises):
                if v>1:
                    logging.info("{}\t{:.2f}%\t1-every-{:.1f}\t{}-noises".format(v-1,100.0*(v-1)/self.total_sents_with_noises,self.total_sentences/(v-1),k))
        
        
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
        
    def __call__(self, l):
        l = l.rstrip()
        self.s = Sentence(self.tokenizer(l))
        self.total_sentences += 1
        self.total_tokens += len(self.s)
        curr_n_noises = 0
        add_n_noises = self.add_n_noises()
        for _ in range(add_n_noises): ### try to add n noises in this sentence
            idx = self.s.get_random_unnoised_idx() #get an unnoised token
            if idx == None:
                break
            #logging.debug("random_unnoised_idx: {}".format(idx))
            for next_noise, _ in self.sort_noises(): ### try all noises sorted by frequency (lower first)
                #logging.debug("next_noise: {}".format(next_noise))
                res = False
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
                elif next_noise == 'spurious':
                    res = self.spurious(idx,self.S)
                if res:
                    self.noises_seen[next_noise] += 1
                    curr_n_noises += 1
                    break
        logging.debug('added {} out of {} noises'.format(curr_n_noises,add_n_noises))
        self.total_sents_with_n_noises[curr_n_noises] += 1
        self.total_noises += curr_n_noises
        self.total_sents_with_noises += 1 if curr_n_noises else 0
        return self.tokenizer.detokenize(self.s()), self.s(triplet=True)

    def misspell(self, idx): #deux
        w = self.s.words[idx]
        if w.is_noisy or not w.is_word():
            return False
        w_txt = w.txt[0]
        txt, err = self.M(w_txt)
        if txt == '' or err == '':
            return False
        self.s.words[idx] = Word([txt],[err],[w_txt],True) #deuc|SUB|deux
        logging.debug('NOISE={}\t{}\t{}\t{}'.format(err,self.s.words[idx].txt,self.s.words[idx].error_type,self.s.words[idx].word_to_predict))
        return True

    def replace(self, idx, R): #avais
        w = self.s.words[idx]
        if w.is_noisy or not w.is_word():
            return False
        w_txt = w.txt[0]
        txt, err = R(w_txt)
        if txt == '' or err == '':
            return False
        self.s.words[idx] = Word([txt],[err],[w_txt],True) #avait|SUB|avais
        logging.debug('NOISE={}\t{}\t{}\t{}'.format(err,self.s.words[idx].txt,self.s.words[idx].error_type,self.s.words[idx].word_to_predict))
        return True

    def punctuation(self, idx): #￭, (may have joiner)
        err = 'SUB:P'
        w = self.s.words[idx]
        if w.is_noisy or not w.is_punc(): #one single punctuation with (or without) joiners
            return False
        w_txt = w.txt[0]
        p = w_txt[1] if w_txt[0] == JOINER else w_txt[0]
        other_p = [x for x in PUNC if x!=p]
        random.shuffle(other_p)
        txt = (JOINER if w_txt[0] == JOINER else '') + other_p[0] + (JOINER if w_txt[-1] == JOINER else '')
        self.s.words[idx] = Word([txt], [err], [w_txt], True) #￭.|SUB|￭,
        logging.debug('NOISE=PUNCTUATION\t{}\t{}\t{}'.format(self.s.words[idx].txt,self.s.words[idx].error_type,self.s.words[idx].word_to_predict))
        return True
    
    def copy(self, idx): #dans
        w = self.s.words[idx]
        if w.is_noisy or not w.is_word():
            return False
        w_txt = w.txt[0]
        self.s.words[idx] = Word([w_txt,w_txt], [NONE,'DEL'], [NONE,NONE], True) #dans|NONE|NONE dans|DEL|NONE
        logging.debug('NOISE=COPY\t{}\t{}\t{}'.format(self.s.words[idx].txt,self.s.words[idx].error_type,self.s.words[idx].word_to_predict))
        return True 

    def case(self, idx):
        w = self.s.words[idx]
        if w.is_noisy or not w.is_word():
            return False
        w_txt = w.txt[0]
        if w_txt.isupper(): ################################################## IBM
            err = 'CASE:X'
            if len(w_txt) > 1 and random.random() < 0.5:
                txt = w_txt[0]+w_txt[1:].lower() #Ibm
            else:
                txt = w_txt.lower()              #ibm
        elif w_txt.islower(): ################################################ ibm
            err = 'CASE:x'
            if len(w_txt) > 1 and random.random() < 0.5:
                txt = w_txt[0].upper()+w_txt[1:] #Ibm
            else:
                txt = w_txt.upper()              #IBM
        elif len(w_txt)>1 and w_txt[0].isupper() and w_txt[1:].islower(): #### Ibm
            err = 'CASE:Xx'
            if random.random() < 0.5:
                txt = w_txt.upper()              #IBM
            else:
                txt = w_txt.lower()              #ibm
        else: ################################################################ IbM
            return False
        self.s.words[idx] = Word([txt], [err], [w_txt], True) #ibm|SUB|IBM
        logging.debug('NOISE={}\t{}\t{}\t{}'.format(err,self.s.words[idx].txt,self.s.words[idx].error_type,self.s.words[idx].word_to_predict))
        return True
    
    def space_add(self, idx): #gestion
        w = self.s.words[idx]
        if w.is_noisy or not w.is_word():
            return False
        w_txt = w.txt[0]
        minlen = 3
        if len(w_txt) < 2*minlen: #minimum length of resulting splitted tokens
            return False
        k = random.randint(minlen,len(w_txt)-minlen)
        wprev = w_txt[:k]
        wpost = w_txt[k:]
        self.s.words[idx] = Word([wprev,wpost], ['JOIN',NONE], [NONE,NONE], True) ### gest|JOIN|NONE ion|NONE|NONE
        logging.debug('NOISE=SPACE:ADD\t{}\t{}\t{}'.format(self.s.words[idx].txt,self.s.words[idx].error_type,self.s.words[idx].word_to_predict))
        return True

    def space_del(self, idx): #mon ami
        if idx >= len(self.s.words)-1:
            return False
        w1, w2 = self.s.words[idx], self.s.words[idx+1]
        if w1.is_noisy or not w1.is_word():
            return False
        if w2.is_noisy or not w2.is_word():
            return False
        w1_txt = w1.txt[0]
        w2_txt = w2.txt[0]
        self.s.words[idx] = Word([w1_txt+w2_txt], ['DIV'], [w1_txt], True) ### monami|DIV|mon
        self.s.words.pop(idx+1) #w2 must be deleted from list of words
        logging.debug('NOISE=SPACE:DEL\t{}\t{}\t{}'.format(self.s.words[idx].txt,self.s.words[idx].error_type,self.s.words[idx].word_to_predict))
        return True 

    def hyphen_del(self, idx): #anti - douleur
        if idx >= len(self.s.words)-2:
            return False
        w1, w2, w3 = self.s.words[idx], self.s.words[idx+1], self.s.words[idx+2]
        if w1.is_noisy or not w1.is_word():
            return False
        if w2.is_noisy or w2.txt[0] != JOINER+'-'+JOINER:
            return False
        if w3.is_noisy or not w3.is_word():
            return False
        w1_txt = w1.txt[0]
        w2_txt = w2.txt[0]
        w3_txt = w3.txt[0]
        self.s.words[idx] = Word([w1_txt,w3_txt], ['ADD',NONE], [JOINER+'-'+JOINER,NONE], True) ### anti|ADD|- douleur|NONE|NONE
        self.s.words.pop(idx+1) #w2 must be deleted from list of words
        self.s.words.pop(idx+1) #w3 must be deleted from list of words
        logging.debug('NOISE=HYPHEN:DEL\t{}\t{}\t{}'.format(self.s.words[idx].txt,self.s.words[idx].error_type,self.s.words[idx].word_to_predict))
        return True

    def hyphen_add(self, idx): #grands magasins
        if idx >= len(self.s.words)-1:
            return False
        w1, w2 = self.s.words[idx], self.s.words[idx+1]
        if w1.is_noisy or not w1.is_word():
            return False
        if w2.is_noisy or not w2.is_word():
            return False
        w1_txt = w1.txt[0]
        w2_txt = w2.txt[0]
        min_words_len = 4
        if len(w1_txt) < min_words_len or len(w2_txt) < min_words_len:
            return False
        self.s.words[idx] = Word([w1_txt, JOINER+'-'+JOINER, w2_txt], [NONE,'DEL',NONE], [NONE,NONE,NONE], True) ### grands|KEEP|NONE #-#|DEL|NONE magasins|KEEP|NONE
        self.s.words.pop(idx+1) #w2 must be deleted from list of words
        logging.debug('NOISE=HYPHEN:ADD\t{}\t{}\t{}'.format(self.s.words[idx].txt,self.s.words[idx].error_type,self.s.words[idx].word_to_predict))
        return True

    def swap(self, idx):
        if idx < 0 or idx >= len(self.s.words)-1:
            return False
        w1, w2 = self.s.words[idx], self.s.words[idx+1]
        if w1.is_noisy or not w1.is_word():
            return False
        if w2.is_noisy or not w2.is_word():
            return False
        w1_txt = w1.txt[0]
        w2_txt = w2.txt[0]
        self.s.words[idx] = Word([w2_txt, w1_txt], ['SWAP', NONE], [NONE, NONE], True)
        self.s.words.pop(idx+1) #w2 must be deleted from list of words
        logging.debug('NOISE=SWAP\t{}\t{}\t{}'.format(self.s.words[idx].txt,self.s.words[idx].error_type,self.s.words[idx].word_to_predict))
        return True

    def spurious(self, idx, S): #add a word after idx
        w = self.s.words[idx]
        if w.is_noisy:
            return False
        w_txt = S()
        w = Word([w_txt], ['DEL'], [NONE], True)
        self.s.words.insert(idx+1,w)
        logging.debug('NOISE=SPURIOUS\t{}\t{}\t{}'.format(self.s.words[idx+1].txt,self.s.words[idx+1].error_type,self.s.words[idx+1].word_to_predict))
        return True
    
###################################################################################################
### MAIN ##########################################################################################
###################################################################################################

if __name__ == '__main__':
    example = {"inflect": "resources/Morphalou3.1_CSV.csv.inflect", "homophone": "resources/Morphalou3.1_CSV.csv.homophone", "spurious": "resources/Morphalou3.1_CSV.csv.spurious", "misspell": {"delete": 1, "repeat": 1, "close": 1, "swap": 1, "diacritics": 10, "consd": 100, "phone": 100}, "noises": {"inflect": 50, "homophone": 50, "punctuation": 1000, "hyphen_add": 100, "hyphen_del": 10, "misspell": 100, "case": 100, "space_add": 100, "space_del": 100, "copy": 1000, "swap": 200, "spurious": 1000}, "max_r": 0.5, "max_n": 10, "seed": 23}
    parser = argparse.ArgumentParser(description="Tool to noise clean text following a noiser configuration file. Example: {}".format(example))
    parser.add_argument('config', type=str, help='noiser configuration file')
    parser.add_argument('-o', type=str, default=None, help='output prefix file')
    parser.add_argument("--debug", action='store_true', help="logging level=DEBUG (INFO)")
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO' if not args.debug else 'DEBUG', None), filename=args.o+'_log' if args.o is not None else None, filemode = 'w')
    with open(args.config,'r') as fd:
        config = json.load(fd)
    random.seed(config['seed'])

    if args.o is not None:
        fd_snt = open(args.o+'_snt', 'w')
        fd_tok = open(args.o+'_tok', 'w')
        
    tic = time.time()
    n = Noiser(Dict2Class(config))
    for l in sys.stdin:
        noised, triplets = n(l)
        if args.o is not None:
            fd_snt.write("{}\n".format(noised))
            fd_tok.write("{}\n".format(triplets))
        else:
            print("{}\t{}".format(noised,triplets))
    toc = time.time()

    if args.o is not None:
        fd_snt.close()
        fd_tok.close()
        
    n.stats(toc-tic)

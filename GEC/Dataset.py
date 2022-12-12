# -*- coding: utf-8 -*-

import sys
import os
import torch
import json
import logging
import numpy as np
from collections import defaultdict
#from model.Vocab import Vocab
from utils.Utils import create_logger, SEPAR1, SEPAR2, KEEP #, conll
from utils.Conll import Conll

#def debug_batch(idxs, batch_raw, batch_ids_src, batch_ids_agg, batch_ids_err, batch_ids_lng, batch_ids_cor, batch_ids_COR):
#    logging.debug('Batch {}'.format(idxs))
#    src = batch_ids_src.tolist()
#    agg = batch_ids_agg.tolist()
#    err = batch_ids_err.tolist()
#    cor = batch_ids_cor.tolist()
#    for k in range(len(idxs)):
#        logging.debug("{} raw: {}".format(idxs[k], batch_raw[k]))
#        logging.debug("{} src [{}]: {}".format(idxs[k], len(src[k]), src[k]))
#        logging.debug("{} agg [{}]: {}".format(idxs[k], len(agg[k]), agg[k]))
#        logging.debug("{} err [{}]: {}".format(idxs[k], len(err[k]), err[k]))
#        logging.debug("{} cor [{}]: {}".format(idxs[k], len(cor[k]), cor[k]))

def pad_listoflists(ll, pad=0, maxl=0, n_subt=1):
    #logging.info('maxl: {}'.format(maxl))
    if maxl==0:
        maxl = max([len(l) for l in ll])
    #logging.info('maxl: {}'.format(maxl))
    for i in range(len(ll)):
        if len(ll[i]) > maxl:
            #logging.error('Bad input data ll: {}'.format(ll))
            sys.exit()
        if isinstance(pad, int) and pad == -1: ### use ll[i][-1]+1 for indexs of coo
            while len(ll[i]) < maxl:
                ll[i].append(ll[i][-1]+1)
        else: ### fill the remaining tokens using pad
            ll[i] += [pad] * (maxl-len(ll[i]))
        #logging.info('ll[{}]: {}'.format(i,ll[i]))
    return torch.Tensor(ll).to(dtype=torch.long) ### convert to tensor
            
class Dataset():
    def __init__(self, fname, err, cor, lng, sha, inl, flauberttok, noiser, args):
        super(Dataset, self).__init__()
        if not os.path.isfile(fname):
            logging.error('Cannot read file {}'.format(fname))
            sys.exit()
        self.err = err
        self.cor = cor #may be None
        self.lng = lng #may be None
        self.sha = sha #may be None
        self.inl = inl #may be None
        #self.flauberttok = flauberttok
        #self.noiser = noiser
        self.args = args
        #
        self.idx_BOS_src = flauberttok.idx_BOS # <s> in encoder
        self.idx_EOS_src = flauberttok.idx_EOS # </s> in encoder
        self.idx_PAD_src = flauberttok.idx_PAD # <pad> in encoder
        #
        self.str_BOS_src = flauberttok.str_BOS # <s> in encoder
        self.str_EOS_src = flauberttok.str_EOS # </s> in encoder
        self.str_PAD_src = flauberttok.str_PAD # <pad> in encoder
        #
        self.idx_PAD_err = err.idx_PAD # <pad> in err vocabulary
        #
        self.idx_PAD_cor = cor.idx_PAD if cor is not None else None #<pad> in cor vocabulary 
        self.idx_PAD_lng = lng.idx_PAD if lng is not None else None #<pad> in lng vocabulary 
        self.idx_PAD_COR = 2 #"<pad>": 2, https://huggingface.co/flaubert/flaubert_base_cased/raw/main/vocab.json
        self.idx_PAD_sha = sha.idx_PAD if sha is not None else None #<pad> in sha vocabulary 
        self.idx_PAD_inl = inl.idx_PAD if inl is not None else None #<pad> in inl vocabulary 
        n_truncated = 0
        n_filtered = 0
        self.Data = []
        conll = Conll(tags=True)
        with open(fname, 'r') as fd:
            idx = 0
            for l in fd:
                ldict = json.loads(l)
                #logging.debug("RAW({}): \n{}".format(idx,conll(ldict)))
                ### inject noise ###
                if noiser is not None:
                    ldict = noiser(ldict)
                    logging.debug("NOISY({}): \n{}".format(idx,conll(ldict)))
                ### truncate if long ###
                if self.args.max_length > 0 and len(ldict) > self.args.max_length:
                    n_truncated += 1
                    ldict = ldict[:self.args.max_length]
                ### filter if empty ###
                if len(ldict) == 0:
                    n_filtered += 1
                    logging.warning('line {} filtered'.format(idx))
                    continue
                ### format sentence ###
                ids_src, str_src, ids_err, ids_cor, ids_lng, ids_COR, ids_sha, ids_agg = self.format_sentence(ldict,idx)
                ### keep record ###
                self.Data.append({'idx':idx, 'ldict':ldict, 'ids_src':ids_src, 'str_src':str_src, 'ids_err':ids_err, 'ids_cor':ids_cor, 'ids_lng':ids_lng, 'ids_COR':ids_COR, 'ids_sha':ids_sha, 'ids_agg':ids_agg})
                idx += 1
        logging.info('Read {} examples from {} [{} filtered, {} truncated]'.format(len(self.Data),fname,n_filtered,n_truncated))
        
    def format_sentence(self, ldict, idx):
        ltoks = []
        lerrors = []
        lcorrs = []
        llngs = []
        #
        str_src = [] # <s>   This  is    my    exxample  </s>
        ids_src = [] # 0     234   31    67    35   678  1     (0:<s>, 1:</s>)
        ids_agg = [] # 0     1     2     3     4    4    5     (indicates to which tok is aligned each subtoken) ### used for inference
        ids_err = [] # 0     1     1     1     4         0     (0:<PAD>, 1:$KEEP, 4:$SPELL)
        ids_cor = [] # 0     0     0     0     624       0     (0:<PAD>, 624:example)
        ids_lng = [] # 0     0     3     0     0         0     (0:<PAD>, 3:pres_ind_3ps)
        ids_COR = [] # [0,0] [0,0] [0,0] [0,0] [347,4]   [0,0] (0:<PAD>, [347,4]:example, when n_subt=2)
        ids_sha = [] # 0     3     2     2     2         0     (0:<PAD>, 3:Xx, 2:x)
        #
        str_src.append('<s>')
        ids_src.append(self.idx_BOS_src)
        ids_agg.append(0)
        ids_err.append(self.idx_PAD_err)
        ids_cor.append(self.idx_PAD_cor)
        ids_lng.append(self.idx_PAD_lng)
        ids_sha.append(self.idx_PAD_sha)
        ids_COR.append([self.idx_PAD_COR]*self.args.n_subt)
        for i,dtok in enumerate(ldict):
            str_src.append(dtok['raw'])
            n_subtok = len(dtok['iraw'])
            ids_src += dtok['iraw']
            ids_agg += [i+1]*n_subtok #ex: [1,1]
            ids_err.append(dtok['ierr'] if 'ierr' in dtok else self.idx_PAD_err)
            ids_cor.append(dtok['icor'] if 'icor' in dtok else self.idx_PAD_cor)
            ids_lng.append(dtok['ilng'] if 'ilng' in dtok else self.idx_PAD_lng)
            ids_COR.append(dtok['iCOR'] if 'iCOR' in dtok else [self.idx_PAD_COR]*self.args.n_subt) ### idx_PAD_COR indicates nothing to predict
            if len(ids_COR[-1]) < self.args.n_subt:
                ids_COR[-1] += [4]* (self.args.n_subt - len(ids_COR[-1])) ### 4 corresponds to flaubert model token: <special0> that must be predicted (while <pad> indicates nothing to predict)
            elif len(ids_COR[-1]) > self.args.n_subt:
                ids_COR[-1] = ids_COR[-1][:self.args.n_subt] ### must be exactly n_subt tokens
            ids_sha.append(dtok['ishp'] if 'ishp' in dtok else self.idx_PAD_sha)
            ### for debug purpose
            if ids_err[-1] != self.idx_PAD_err:
                ltoks.append(dtok['raw'])
                lerrors.append(self.err[ids_err[-1]])
            if ids_cor[-1] != self.idx_PAD_cor:
                lcorrs.append(self.cor[ids_cor[-1]])
            if ids_lng[-1] != self.idx_PAD_lng:
                llngs.append(self.lng[ids_lng[-1]])
            ###
        str_src.append('</s>')
        ids_src.append(self.idx_EOS_src)
        ids_agg.append(ids_agg[-1]+1)
        ids_err.append(self.idx_PAD_err)
        ids_cor.append(self.idx_PAD_cor)
        ids_lng.append(self.idx_PAD_lng)
        ids_COR.append([self.idx_PAD_COR]*self.args.n_subt)
        ids_sha.append(self.idx_PAD_sha)
        assert(len(ids_src) == len(ids_agg))
        assert(len(str_src) == len(ids_err))
        assert(len(str_src) == len(ids_cor))
        assert(len(str_src) == len(ids_lng))
        assert(len(str_src) == len(ids_COR))
        assert(len(str_src) == len(ids_sha))
        #logging.debug('idx {}'.format(idx))
        #logging.debug('ids_src {}'.format(ids_src))
        #logging.debug('ids_agg {}'.format(ids_agg))
        #logging.debug('str_src {}\t{}'.format(str_src, '\t'.join(ltoks)))
        #logging.debug('ids_err {}\t{}'.format(ids_err, '\t'.join(lerrors)))
        #logging.debug('ids_cor {}\t{}'.format(ids_cor, '\t'.join(lcorrs)))
        #logging.debug('ids_lng {}\t{}'.format(ids_lng, '\t'.join(llngs)))
        #logging.debug('ids_COR {}'.format(ids_COR))
        #logging.debug('ids_sha {}'.format(ids_sha))
        return ids_src, str_src, ids_err, ids_cor, ids_lng, ids_COR, ids_sha, ids_agg

#    def build_cor_aggregate(self, field_cor):
#        myids_cor = list(map(int,field_cor.split(SEPAR2)))
#        if len(myids_cor) < self.args.n_subt:
#            myids_cor += [4]*(self.args.n_subt-len(myids_cor)) #4 corresponds to flaubert model token: <special0> that must be predicted (PAD is not)
#        elif len(myids_cor) > self.args.n_subt:
#            myids_cor = myids_cor[:self.args.n_subt] #truncation... problem if the right word (using n subtokens) cannot be produced
#        return [myids_cor]
            
#    def build_err_gector(self,field_err,l):
#        myids_err = self.err[field_err]
#        lerr = [self.idx_PAD_err] * l
#        if self.args.aggreg == 'first':
#            lerr[0] = myids_err
#        else:
#            lerr[-1] = myids_err
#        return lerr
        
#    def build_cor_gector(self,fields,n_subtok):
#        ### n_subtok is the number of subtokens of current source words
#        ### self.args.n_subt is the number of subtokens to be predicted as correction
#        if len(fields)<7:
#            myids_cor = [self.idx_PAD_cor]*self.args.n_subt
#        else:
#            myids_cor = list(map(int,fields[6].split(SEPAR2)))
#            
#        if len(myids_cor) < self.args.n_subt:
#            myids_cor += [4]*(self.args.n_subt-len(myids_cor)) #4 corresponds to flaubert model token: <special0> that must be predicted (PAD is not)
#        elif len(myids_cor) > self.args.n_subt:
#            myids_cor = myids_cor[:self.args.n_subt]
#
#        pad_cor = [self.idx_PAD_cor] * self.args.n_subt ### a word correction is formed of n_subtoken <pad>'s
#        lcor = [pad_cor] * n_subtok ### initially all subtokens of current word are padded
#        ### exx   ample
#        ### [0,0] [0,0]
#        if self.args.aggreg == 'first':
#            lcor[0] = myids_cor
#            ### exx   ample  #n_subtok=2
#            ### [15,0] [0,0] #when n_subt=1 (aggreg is first)
#        else:
#            lcor[-1] = myids_cor
#            ### exx   ample  #n_subtok=2
#            ### [0,0] [15,0] #when n_subt=1 (aggreg is last)
#        return lcor
    
    def __len__(self):
        return len(self.Data)

    def __iter__(self):
        assert len(self.Data) > 0, 'Empty dataset'
        logging.info('Shuffling dataset to build shards')
        idx_Data = [i for i in range(len(self.Data))]
        np.random.shuffle(idx_Data)
        self.args.shard_size = self.args.shard_size or len(idx_Data)
        shards = [idx_Data[i:i+self.args.shard_size] for i in range(0, len(idx_Data), self.args.shard_size)] # split dataset in shards
        logging.info('Built {} shards with up to {} examples'.format(len(shards),self.args.shard_size))
        for s,shard in enumerate(shards):
            logging.info('Building batchs for shard {}/{}'.format(s+1,len(shards)))
            batchs = self.build_batchs(shard)
            logging.info('Found {} batchs'.format(len(batchs)))
            for batch in batchs:
                yield self.format_batch(batch)
            logging.info('End of shard {}/{}'.format(s+1,len(shards)))
        logging.info('End of dataset')
            
    def build_batchs(self, shard):
        shard_len = [len(self.Data[idx]['ids_src']) for idx in shard]
        shard = np.asarray(shard)
        ord_lens = np.argsort(shard_len) #sort by lens (lower to higher lengths)
        shard = shard[ord_lens] #examples in shard are now sorted by lens
        batchs = [] ### build batchs of same (similar) size
        curr_batch = []
        curr_batch_len = 0
        for idx in shard:
            if curr_batch_len + self.len_example(idx) > self.args.batch_size:
                if curr_batch_len:
                    batchs.append(curr_batch)
                curr_batch = []
                curr_batch_len = 0
            curr_batch.append(idx)
            curr_batch_len += self.len_example(idx)
        if curr_batch_len:
            batchs.append(curr_batch)
        np.random.shuffle(batchs)
        return batchs

    def len_example(self, idx):
        if self.args.batch_type == 'tokens':
            return len(self.Data[idx]['ids_src']) ### number of subwords
        return 1 ### number of sentences

        
    def format_batch(self, idxs):
        batch_raw = []
        batch_str_src = []
        batch_ids_src = []
        batch_ids_agg = []
        batch_ids_err = []
        batch_ids_cor = []
        maxl = 0
        for idx in idxs:
            if 'str_src' not in self.Data[idx]:
                logging.warning('filtered {} line'.format(idx))
                continue
            batch_raw.append(self.Data[idx]['raw'])
            if maxl < len(self.Data[idx]['ids_src']):
                maxl = len(self.Data[idx]['ids_src'])
            batch_str_src.append(self.Data[idx]['str_src'])
            batch_ids_src.append(self.Data[idx]['ids_src'])
            batch_ids_agg.append(self.Data[idx]['ids_agg'])
            batch_ids_err.append(self.Data[idx]['ids_err'])
            batch_ids_cor.append(self.Data[idx]['ids_cor'])
        ### convert to tensor
        batch_ids_src = pad_listoflists(batch_ids_src,pad=self.idx_PAD_src,maxl=maxl)
        batch_ids_agg = pad_listoflists(batch_ids_agg,pad=-1,maxl=maxl)
        #if ids_err/ids_cor are smaller than ids_src/ids_agg i add PAD so as to obtain the same size
        batch_ids_err = pad_listoflists(batch_ids_err,pad=self.idx_PAD_err,maxl=maxl)
        batch_ids_cor = pad_listoflists(batch_ids_cor,pad=[self.idx_PAD_cor]*self.args.n_subt,maxl=maxl,n_subt=self.args.n_subt)
#        if self.args.log == 'debug':
#            debug_batch(idxs, batch_raw, batch_ids_src, batch_ids_agg, batch_ids_err, batch_ids_cor)
        return [batch_ids_src, batch_ids_agg, batch_ids_err, batch_ids_cor, idxs, batch_str_src]

    
    def reformat_batch(self, batch):
        batch_words = []
        batch_ids = []
        batch_ids2words = []
        maxl_ids = 0
        for i in range(len(batch)):
            ids = self.flauberttok.ids(batch[i], add_special_tokens=True, is_split_into_words=False)
            words, ids2words, _, _ = self.flauberttok.words_ids2words_subwords_lids(ids)
            #logging.debug('reformat {}'.format(batch[i]))
            #logging.debug('words {}'.format(words))
            #logging.debug('ids {}'.format(ids))
            #logging.debug('ids2words {}'.format(ids2words))
            batch_words.append(words)
            batch_ids.append(ids)
            batch_ids2words.append(ids2words)
            maxl_ids = len(ids) if len(ids) > maxl_ids else maxl_ids
        batch_ids = pad_listoflists(batch_ids, pad=self.idx_PAD_src, maxl=maxl_ids)
        batch_ids2words = pad_listoflists(batch_ids2words, pad=-1, maxl=maxl_ids)
        return [batch_ids, batch_ids2words, batch_words]



# -*- coding: utf-8 -*-

import sys
import os
import torch
import logging
import numpy as np
from collections import defaultdict

NONE = 'NONE'
#from model.Vocab import Vocab
#from utils.Utils import create_logger, SEPAR1, SEPAR2, KEEP #, conll
#from utils.Conll import Conll

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

def pad_listoflists(ll, pad=0, maxl=0):
    if maxl==0:
        maxl = max([len(l) for l in ll])
    for i in range(len(ll)):
        if len(ll[i]) > maxl:
            sys.exit()
        if isinstance(pad, int) and pad == -1: ### use ll[i][-1]+1 for indexs of coo
            while len(ll[i]) < maxl:
                ll[i].append(ll[i][-1]+1)
        else: ### fill the remaining tokens using pad
            ll[i] += [pad] * (maxl-len(ll[i]))
    return torch.Tensor(ll).to(dtype=torch.long) ### convert to tensor
            
class Dataset():
    def __init__(self, fname, err, token, args):
        super(Dataset, self).__init__()
        self.err = err
        self.token = token
        self.n_subt = args.n_subtok
        self.args = args
        #self.token.unk_token_id, self.token.unk_token # <unk> in encoder
        #self.token.bos_token_id, self.token.bos_token # <s> in encoder
        #self.token.eos_token_id, self.token.eos_token # </s> in encoder
        #self.token.pad_token_id, self.token.pad_token # <pad> in encoder
        #self.err.idx_PAD # <pad> in err vocabulary
        #self.err.idx_UNK # <unk> in err vocabulary
        n_truncated = 0
        n_filtered = 0
        self.Data = []
        idx = 0
        for f in fname.split(','):
            logging.info('Reading {}'.format(f))
            with open(f, 'r') as fd:
                for l in fd:
                    l = l.rstrip()
                    tok = l.split('\t')
                    if len(tok) == 3:
                        ldict = list(eval(tok[2]))
                    elif len(tok) == 1:
                        pass
                    else:
                        logging.error('cannot parse {} input line: {}'.format(idx,l))
                        sys.exit()

                    if len(ldict) == 0:
                        n_filtered += 1
                        logging.warning('empty line {} filtered'.format(idx))
                        continue
                    elif self.args.max_len > 0 and len(ldict) >= self.args.max_len:
                        n_truncated += 1
                        ldict = ldict[:self.args.max_len-1]
                    ids_src, ids_agg, ids_err, ids_cor = self.format_ldict(ldict)
                    #logging.debug("IDX:{}\nSRC:{}\nAGG:{}\nERR:{}\nCOR:{}".format(idx, ids_src, ids_agg, ids_err, ids_cor))
                    self.Data.append({'idx':idx, 'ids_src':ids_src, 'ids_agg':ids_agg, 'ids_err':ids_err, 'ids_cor':ids_cor}) #, 'ldict':ldict})
                    idx += 1
        logging.info('Read {} examples from {} [{} filtered, {} truncated]'.format(len(self.Data),fname,n_filtered,n_truncated))

    def format_ldict(self, ldict):
                     #This  is    my    exxample  </s>
        ids_src = [] #234,  31,   67,   35, 678,  1
        ids_agg = [] #0,    1,    2,    3,  3,    4
        ids_err = [] #2,    2,    2,    5,              (0:<pad>, 1:<unk>, 2:NONE, 5:SUB:M)
        ids_cor = [] #[0,0] [0,0] [0,0] [347,4]         (0:<pad>, [347,4]:example, when n_subt=2)
        for i in range(len(ldict)):
            ids = ldict[i]['i']
            err = 'NONE' if ldict[i]['e'] is None else ldict[i]['e']['e']
            cor = [self.token.pad_token_id] if ldict[i]['e'] is None or ldict[i]['e']['i'] is None else ldict[i]['e']['i']+[self.token.eos_token_id]
            #cor = [self.token.pad_token_id]*self.n_subt if ldict[i]['e'] is None or ldict[i]['e']['i'] is None else ldict[i]['e']['i']+[self.token.eos_token_id]*(self.n_subt-len(ldict[i]['e']['i']))
            ids_src += ids
            ids_agg += [i]*len(ids)
            ids_err.append(self.err[err])
            ids_cor.append(cor)
            if len(ids_cor[-1]) < self.n_subt: ### add padding
                ids_cor[-1] += [self.token.pad_token_id]*(self.n_subt-len(ids_cor[-1]))
            elif len(ids_cor[-1]) > self.n_subt: ### trunc to n_subt
                logging.warning('truncated token correction: {} (len={} {})'.format(ldict[i]['e']['t'],len(ids_cor[-1]),ids_cor[-1]))
                ids_cor[-1] = ids_cor[-1][:self.n_subt] #or ids_cor[-1] = [self.token.pad_token_id] * self.n_subt
        #</s> added to properly call the encoder
        ids_src.append(self.token.eos_token_id)
        ids_agg.append(ids_agg[-1]+1)
#        ids_err.append(self.err.idx_PAD)
#        ids_cor.append([self.token.pad_token_id]*self.n_subt)            
        assert(len(ids_src) == len(ids_agg))
        assert(len(ids_cor) == len(ids_err))
        assert(ids_agg[-1] == len(ids_cor))
        return ids_src, ids_agg, ids_err, ids_cor

    
    def __len__(self):
        return len(self.Data)

    def __iter__(self):
        assert len(self.Data) > 0, 'Empty dataset'
        logging.info('Shuffling dataset to build shards')
        idx_Data = [i for i in range(len(self.Data))]
        np.random.shuffle(idx_Data)
        self.args.shard_sz = self.args.shard_sz or len(idx_Data)
        shards = [idx_Data[i:i+self.args.shard_sz] for i in range(0, len(idx_Data), self.args.shard_sz)] # split dataset in shards
        logging.info('Built {} shards with up to {} examples'.format(len(shards),self.args.shard_sz))
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
            if curr_batch_len + self.len_example(idx) > self.args.batch_sz:
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
        if self.args.batch_tp == 'tokens':
            return len(self.Data[idx]['ids_src']) ### number of subwords
        return 1 ### number of sentences

        
    def format_batch(self, idxs):
        batch_ids_src = []
        batch_ids_agg = []
        batch_ids_err = []
        batch_ids_cor = []
        maxl1 = 0
        maxl2 = 0
        pad_cor = [self.token.pad_token_id]*self.args.n_subtok
        for idx in idxs:
            maxl1 = max(maxl1, len(self.Data[idx]['ids_src']))
            maxl2 = max(maxl2, len(self.Data[idx]['ids_err']))
            batch_ids_src.append(self.Data[idx]['ids_src'])
            batch_ids_agg.append(self.Data[idx]['ids_agg'])
            batch_ids_err.append(self.Data[idx]['ids_err'])
            batch_ids_cor.append(self.Data[idx]['ids_cor'])
        ### convert to tensor
        batch_ids_src = pad_listoflists(batch_ids_src,pad=self.token.pad_token_id,maxl=maxl1)
        batch_ids_agg = pad_listoflists(batch_ids_agg,pad=-1,maxl=maxl1)
        batch_ids_err = pad_listoflists(batch_ids_err,pad=self.err.idx_PAD,maxl=maxl2)
        batch_ids_cor = pad_listoflists(batch_ids_cor,pad=pad_cor,maxl=maxl2)
        return [batch_ids_src, batch_ids_agg, batch_ids_err, batch_ids_cor, idxs]

    
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



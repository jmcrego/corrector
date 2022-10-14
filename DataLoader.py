import torch
import logging
import argparse
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class DataLoader():

    def __init__(self, args, tokenizer, fsrc, ftgt):
        self.args = args
        self.tokenizer = tokenizer
        self.source_raw = []
        self.target_raw = []
        self.n_batchs_retrieved = 0
        
        if fsrc is not None:
            for fsrc in fsrc.split(','):
                with open(fsrc,'r') as fd:
                    lsrc = [args.prefix+l.rstrip() for l in fd]
            self.source_raw.extend(lsrc)
            logging.info('Read {} source sentences from {}'.format(len(self.source_raw),fsrc))
            
        if ftgt is not None:
            for ftgt in ftgt.split(','):
                with open(ftgt,'r') as fd:
                    ltgt = [l.rstrip() for l in fd]
            self.target_raw.extend(ltgt)
            logging.info('Read {} target sentences from {}'.format(len(self.target_raw),ftgt))

        assert len(self.target_raw)==0 or len(self.source_raw) == len(self.target_raw)

    def __len__(self):
        return len(self.source_raw)
        
    def __iter__(self):
        for shard_ini in range(0,len(self.source_raw), self.args.shard_sz):
            shard_fin = min(shard_ini+self.args.shard_sz,len(self.source_raw))
            shard_source_raw = self.source_raw[shard_ini:shard_fin]
            self.shard_source_dic = self.tokenizer(shard_source_raw, max_length=self.args.maxl_src, truncation=True)
            logging.info('tokenized source shard with {} examples'.format(len(shard_source_raw)))
            if len(self.target_raw):
                shard_target_raw = self.target_raw[shard_ini:shard_fin]
                self.shard_target_dic = self.tokenizer(shard_target_raw, max_length=self.args.maxl_tgt, truncation=True)
                logging.info('tokenized target shard with {} examples'.format(len(shard_target_raw)))

            self.indexs_source_sorted_by_len = np.argsort([len(l) for l in self.shard_source_dic['input_ids']]).tolist()
        
            batchs = []
            while len(self.indexs_source_sorted_by_len):
                batchs.append(self.get_batch_from_shard(shard_ini))
            logging.info('built {} batchs'.format(len(batchs)))

            np.random.shuffle(batchs)
            for batch in batchs:
                self.n_batchs_retrieved += 1
                yield batch

            
    def get_batch_from_shard(self, shard_ini):
        batch_source_ids = []
        batch_source_msk = []
        batch_target_ids = []
        batch_target_msk = []
        batch_indexs = []
        while len(self.indexs_source_sorted_by_len):
            index = self.indexs_source_sorted_by_len[0]
            source_ids = self.shard_source_dic['input_ids'][index]
            source_msk = self.shard_source_dic['attention_mask'][index]
            max_len = len(source_ids) #length is growing
            total_sentences = len(batch_source_ids) + 1
            total_tokens = max_len * total_sentences
            if total_sentences == 1 or (self.args.batch_tp == 'tokens' and total_tokens <= self.args.batch_sz) or (self.args.batch_tp == 'sentences' and total_sentences <= self.args.batch_sz):
                self.indexs_source_sorted_by_len.pop(0)
                batch_indexs.append(index + shard_ini)
                batch_source_ids.append(source_ids)
                batch_source_msk.append(source_msk)
                if len(self.target_raw):
                    target_ids = self.shard_target_dic['input_ids'][index]
                    target_msk = self.shard_target_dic['attention_mask'][index]
                    batch_target_ids.append(target_ids)
                    batch_target_msk.append(target_msk)
            else:
                return self.pad_batch_LongTensor(batch_source_ids, batch_source_msk, batch_target_ids, batch_target_msk, batch_indexs)

        if len(batch_source_ids):
            return self.pad_batch_LongTensor(batch_source_ids, batch_source_msk, batch_target_ids, batch_target_msk, batch_indexs)

        
    def pad_batch_LongTensor(self, batch_source_ids, batch_source_msk, batch_target_ids, batch_target_msk, batch_indexs):
        batch_source_ids  = pad_sequence([torch.LongTensor(l) for l in batch_source_ids], batch_first=True, padding_value=0).to(dtype=torch.long)
        batch_source_msk  = pad_sequence([torch.LongTensor(l) for l in batch_source_msk], batch_first=True, padding_value=0).to(dtype=torch.long)
        if len(batch_target_ids) == 0:
            return { 'source_ids': batch_source_ids, 'source_mask': batch_source_msk, 'indexs':batch_indexs }
        batch_target_ids  = pad_sequence([torch.LongTensor(l) for l in batch_target_ids], batch_first=True, padding_value=0).to(dtype=torch.long)
        batch_target_msk  = pad_sequence([torch.LongTensor(l) for l in batch_target_msk], batch_first=True, padding_value=0).to(dtype=torch.long)
        return { 'source_ids': batch_source_ids, 'source_mask': batch_source_msk, 'target_ids': batch_target_ids, 'target_mask': batch_target_msk, 'indexs':batch_indexs }
        
        
if __name__ == "__main__":

    import sys
    from transformers import T5Tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None, type=str, help="source file")
    parser.add_argument("--tgt", default=None, type=str, help="target file")
    parser.add_argument("--prefix", default="GEC: ", type=str, help="prefix (GEC: )")
    parser.add_argument("--maxl_src", default=200, type=int, help="max source length (200)")
    parser.add_argument("--maxl_tgt", default=200, type=int, help="max target length (200)")
    parser.add_argument("--batch_sz", default=1024, type=int, help="batch size (1024)")
    parser.add_argument("--batch_tp", default="tokens", type=str, help="batch type: sentences OR tokens (tokens)")
    parser.add_argument("--shard_sz", default=500000, type=int, help="shard size (500000)")
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO', None))

    tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=args.maxl_src)
    dl = DataLoader(args, tokenizer, args.src, args.tgt)
    for i,batch in enumerate(dl):
        if i % 5000 == 0:
            print('{}'.format(batch))


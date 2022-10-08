import torch
import logging
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader#, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence

def file_to_list(f, prefix=''):
    if f is None:
        return None
    with open(f,'r') as fd:
        lines = [ prefix+l.rstrip() for l in fd ]
    return lines

def custom_collate(batch):
    source_ids  = [b['source_ids'].clone().detach() for b in batch]
    source_mask = [b['source_mask'].clone().detach() for b in batch]
    source_ids  = pad_sequence(source_ids, batch_first=True, padding_value=0).to(dtype=torch.long)
    source_mask = pad_sequence(source_mask, batch_first=True, padding_value=0).to(dtype=torch.long)
    if 'target_ids' in batch[0]:
        target_ids  = [b['target_ids'].clone().detach() for b in batch]
        target_mask = [b['target_mask'].clone().detach() for b in batch]
        target_ids  = pad_sequence(target_ids, batch_first=True, padding_value=0).to(dtype=torch.long)
        target_mask = pad_sequence(target_mask, batch_first=True, padding_value=0).to(dtype=torch.long)
        return { 'source_ids': source_ids, 'source_mask': source_mask, 'target_ids': target_ids, 'target_mask': target_mask }        
    return { 'source_ids': source_ids, 'source_mask': source_mask }

class CustomDataset(Dataset):

    def __init__(self, args, tokenizer, fsource, ftarget):
        self.max_source_len = args.maxl_src
        self.max_target_len = args.maxl_tgt
        self.tokenizer = tokenizer
        self.source = file_to_list(fsource, prefix=args.prefix)
        self.target = file_to_list(ftarget) #return None if no ftarget is None
        assert self.target is None or len(self.source) == len(self.target)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_text = " ".join(str(self.source[index]).split())
        source = self.tokenizer([source_text], max_length=self.max_source_len, truncation=True, return_tensors="pt")
        if self.target is None or self.max_target_len is None:
            return {"source_ids":source["input_ids"].squeeze().to(dtype=torch.long),"source_mask":source["attention_mask"].squeeze().to(dtype=torch.long)}

        target_text = " ".join(str(self.target[index]).split())
        target = self.tokenizer([target_text], max_length=self.max_target_len, truncation=True, return_tensors="pt")
        return {"source_ids": source["input_ids"].squeeze().to(dtype=torch.long),"source_mask": source["attention_mask"].squeeze().to(dtype=torch.long),"target_ids": target["input_ids"].squeeze().to(dtype=torch.long),"target_mask": target["attention_mask"].squeeze().to(dtype=torch.long)}

class DataSetLoader():

    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.args = args
    
    def __call__(self, fsrc, ftgt, shuffle=True):
        dataset = CustomDataset(self.args, self.tokenizer, fsrc, ftgt)
        params = { "batch_size": self.args.batch_sz, "shuffle": shuffle, "num_workers": 0, "collate_fn": custom_collate }    
        loader = DataLoader(dataset, **params)
        #loader = DataLoader(dataset, batch_size=self.args.batch_sz, shuffle=shuffle, num_workers=0, collate_fn=custom_collate)
        logging.info("loader contains {} batchs ({} sentences)".format(len(loader),len(dataset)))
        return loader, len(dataset)
    

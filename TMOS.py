import os
import sys
import torch
import logging
from transformers import AdamW, get_scheduler
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import MT5ForConditionalGeneration#, T5Tokenizer
from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

class TMOS():

    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        if os.path.exists(self.args.dir + '/path'): ### resume training or inference
            read_from = self.args.dir
            save_local = False
            with open(self.args.dir + '/path', 'r') as fd:
                self.args.path = fd.readlines()[0].rstrip()
        else: ### begin training
            if self.args.path is None:
                logging.error('impossible to guess path from dir {} use --path'.format(self.args.dir))
                sys.exit()            
            save_local = True
            read_from = self.args.path
            if not os.path.exists(self.args.dir): ### resume training or inference
                os.mkdir(self.args.dir)
            with open(self.args.dir + '/path', 'w') as f: #save path
                f.write(self.args.path)
                
        if self.args.path == 't5-base': #https://huggingface.co/docs/transformers/model_doc/t5 
            logging.info('loading local {} tokenizer/model'.format(read_from))
            self.tokenizer = T5Tokenizer.from_pretrained(read_from)
            self.model = T5ForConditionalGeneration.from_pretrained(read_from)
        elif self.args.path == 'google/mt5-base': #https://huggingface.co/docs/transformers/model_doc/mt5
            logging.info('loading local {} tokenizer/model'.format(read_from))
            self.tokenizer = T5Tokenizer.from_pretrained(read_from)
            self.model = MT5ForConditionalGeneration.from_pretrained(read_from)
        elif self.args.path == 'facebook/mbart-large-50': #https://huggingface.co/docs/transformers/model_doc/mbart
            logging.info('loading local {} tokenizer/model'.format(read_from))
            self.tokenizer = MBart50TokenizerFast.from_pretrained(read_from, src_lang="fr", tgt_lang="fr")
            self.model = MBartForConditionalGeneration.from_pretrained(read_from)    
        elif self.args.path == 'facebook/m2m100_418M': #https://huggingface.co/docs/transformers/model_doc/m2m_100
            logging.info('loading local {} tokenizer/model'.format(read_from))
            self.tokenizer = M2M100Tokenizer.from_pretrained(read_from, src_lang="fr")
            self.model = M2M100ForConditionalGeneration.from_pretrained(read_from)                
        else:
            logging.error('invalid path option {}'.format(self.args.path))
            sys.exit()

        if save_local: #save local
            self.model.save_pretrained(self.args.dir)
            self.tokenizer.save_pretrained(self.args.dir)
            
        self.model = self.model.to(self.device)

    def build_optimizer_scheduler(self, num_training_steps):
        logging.info('build optimizer/scheduler')
        self.optimizer = AdamW(params=self.model.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2), eps=self.args.eps, weight_decay=self.args.wdecay)
        if self.args.sched is not None:
            lr_scheduler = get_scheduler(self.args.sched, optimizer=self.optimizer, num_warmup_steps=self.args.warmup, num_training_steps=num_training_steps)
            self.scheduler = lr_scheduler
            
    def save(self):
        logging.info("saving model...")
        self.model.save_pretrained(self.args.dir) #save model
        #logging.info("saving optimizer...")
        #torch.save(self.optimizer.state_dict(), self.args.dir + "/optimizer.pth.tar") #save optim
                
    def __call__(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)

    
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    
    def generate(self, input_ids, attention_mask, is_inference=False):
        #m2m100_418M use: forced_bos_token_id=tokenizer.get_lang_id("en")
        return self.model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   do_sample=False,
                                   max_length=self.args.maxl_tgt,
                                   num_beams=self.args.beam_sz,
                                   repetition_penalty=self.args.rep_pty,
                                   length_penalty=self.args.len_pty,
                                   early_stopping=self.args.early_stopping,
                                   num_return_sequences=self.args.n_best if is_inference else 1)

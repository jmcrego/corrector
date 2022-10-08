import os
import sys
import torch
import logging
from transformers import AdamW, get_linear_schedule_with_warmup, get_scheduler

class TMO():

    
    def __init__(self, args, num_training_steps):
        self.args = args
        self.num_train_steps = num_training_steps
        if os.path.exists(self.args.dir + '/path'):
            with open(self.args.dir + '/path', 'r') as fd:
                self.args.path = fd.readlines()[0].rstrip()
                logging.info('path: {}'.format(self.args.path))
        if self.args.path is None:
            logging.error('impossible to guess path from dir {} use --path'.format(self.args.dir))
            sys.exit()            

            
    def save(self):
        logging.info("saving model...")
        self.model.save_pretrained(self.args.dir) #save model
        #logging.info("saving optimizer...")
        #torch.save(self.optimizer.state_dict(), self.args.dir + "/optimizer.pth.tar") #save optim
        if not os.path.exists(self.args.dir + '/path'):
            logging.info("saving tokenizer...")
            self.tokenizer.save_pretrained(self.args.dir) #save tokenizer
            with open(self.args.dir + '/path', 'w') as f: #save path
                f.write(self.args.path)

                
    def load(self, device):
        self.device = device

        if not os.path.exists(self.args.dir):
            os.makedirs(self.args.dir)

        ###############################################################################################################
        if self.args.path == 't5-base': #https://huggingface.co/docs/transformers/model_doc/t5 
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            if not os.path.exists(self.args.dir + '/path'): #begin learning
                logging.info('downloading huggingface {} tokenizer/model into {}'.format(self.args.path,self.args.dir))
                self.tokenizer = T5Tokenizer.from_pretrained(self.args.path)
                self.model = T5ForConditionalGeneration.from_pretrained(self.args.path)
            else: #resume learning or inference
                logging.info('loading local {} tokenizer/model'.format(self.args.dir))
                self.tokenizer = T5Tokenizer.from_pretrained(self.args.dir)
                self.model = T5ForConditionalGeneration.from_pretrained(self.args.dir)
                
        ###############################################################################################################
        elif self.args.path == 'google/mt5-base': #https://huggingface.co/docs/transformers/model_doc/mt5
            from transformers import T5Tokenizer, MT5ForConditionalGeneration
            if not os.path.exists(self.args.dir + '/path'): #begin learning
                logging.info('downloading huggingface {} tokenizer/model into {}'.format(self.args.path,self.args.dir)) 
                self.tokenizer = T5Tokenizer.from_pretrained(self.args.path)
                self.model = MT5ForConditionalGeneration.from_pretrained(self.args.path)
            else: #resume learning or inference
                logging.info('loading local {} tokenizer/model'.format(self.args.dir))
                self.tokenizer = T5Tokenizer.from_pretrained(self.args.path)
                self.model = MT5ForConditionalGeneration.from_pretrained(self.args.path)
                
        ###############################################################################################################
        elif self.args.path == 'facebook/mbart-large-50': #https://huggingface.co/docs/transformers/model_doc/mbart
            from transformers import MBartForConditionalGeneration, MBartTokenizer
            if not os.path.exists(self.args.dir + '/path'): #begin learning
                logging.info('downloading huggingface {} tokenizer/model into {}'.format(self.args.path,self.args.dir)) 
                self.tokenizer = MBart50TokenizerFast.from_pretrained(self.args.path)
                self.model = MBartForConditionalGeneration.from_pretrained(self.args.path)
            else: #resume learning or inference
                logging.info('loading local {} tokenizer/model'.format(self.args.dir))
                self.tokenizer = MBart50TokenizerFast.from_pretrained(self.args.path)
                self.model = MBartForConditionalGeneration.from_pretrained(self.args.path)    
                
        ###############################################################################################################
        elif self.args.path == 'facebook/m2m100_418M': #https://huggingface.co/docs/transformers/model_doc/m2m_100
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
            if not os.path.exists(self.args.dir + '/path'): #begin learning
                logging.info('downloading huggingface {} tokenizer/model into {}'.format(self.args.path,self.args.dir)) 
                self.tokenizer = M2M100Tokenizer.from_pretrained(self.args.path, src_lang="fr", tgt_lang="fr")
                self.model = M2M100ForConditionalGeneration.from_pretrained(self.args.path)
            else: #resume learning or inference
                logging.info('loading local {} tokenizer/model'.format(self.args.dir))
                self.tokenizer = M2M100Tokenizer.from_pretrained(self.args.path, src_lang="fr", tgt_lang="fr")
                self.model = M2M100ForConditionalGeneration.from_pretrained(self.args.path)                
                
        ###############################################################################################################
        else:
            logging.error('invalid path option {}'.format(self.args.path))
            sys.exit()
            
        self.model = self.model.to(self.device)

        if num_training_steps: #learning
            logging.info('build optimizer/scheduler')
            self.optimizer = AdamW(params=self.model.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2), eps=self.args.eps, weight_decay=self.args.wdecay)
            self.scheduler = get_scheduler(self.args.scheduler, optimizer=self.optimizer, num_warmup_steps=self.args.warmup, num_training_steps=self.num_training_steps)
            #self.optimizer = torch.optim.AdamW (params=self.model.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2), eps=self.args.eps, weight_decay=self.args.wdecay) ### initialized optimizer
            #if os.path.exists(self.args.dir + "/optimizer.pth.tar"):
            #    logging.info('loading local {} optimizer'.format(self.args.dir))
            #    self.optimizer.load_state_dict(torch.load(self.args.dir + "/optimizer.pth.tar")) #, map_location="cpu")) # load previous values
            if not os.path.exists(self.args.dir + "/path"):
                self.save() ### initial save with default tokenizer/model
                
            
    def __call__(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)

    
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    
    def generate(self, input_ids, attention_mask, is_inference=False):
        return self.model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   do_sample=False,
                                   max_length=self.args.maxl_tgt,
                                   num_beams=self.args.beam_sz,
                                   repetition_penalty=self.args.rep_pty,
                                   length_penalty=self.args.len_pty,
                                   early_stopping=self.args.early_stopping,
                                   num_return_sequences=self.args.n_best if is_inference else 1)


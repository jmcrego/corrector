import os
import sys
import time
import torch
import random
import logging
import argparse
import pyonmttok
import numpy as np
import pandas as pd
import edit_distance
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from wer import wer
from DataSetLoader import DataSetLoader
from TMOS import TMOS

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args, epoch, tmos, train_loader, valid_loader, onmttok):
    N = len(train_loader)
    min_valid_wer = None
    sum_loss = 0.
    n_steps = 0
    for batch in train_loader:
        tmos.model.train()
        input_ids = batch["source_ids"].to(tmos.device, dtype=torch.long)
        attention_mask = batch["source_mask"].to(tmos.device, dtype=torch.long) 
        labels = batch["target_ids"].to(tmos.device, dtype=torch.long)
        labels[labels == tmos.tokenizer.pad_token_id] = -100
        ### forward
        outputs = tmos(input_ids, attention_mask, labels) 
        loss = outputs.loss #[0]
        n_steps += 1
        loss.backward()
        sum_loss += loss
        if args.clip:
            torch.nn.utils.clip_grad_norm_(tmos.model.parameters(), args.clip)
        tmos.optimizer.step()
        if args.sched is not None:
            tmos.scheduler.step()
        tmos.optimizer.zero_grad()

        if n_steps % args.report_n == 0:
            logging.info("Epoch:{}/{} Step:{}/{} loss:{:.6f} lr={:.6f}".format(epoch,args.epochs,n_steps,N,sum_loss/args.report_n,tmos.optimizer.param_groups[0]["lr"]))
            sum_loss = 0.
            
        if n_steps % args.valid_n == 0:
            logging.info("Running validation...")
            wer_score, nhyp, nref, generated_txts = validation(args, tmos, valid_loader, onmttok)
            logging.info("valid wer: {:.2f} (#hyp={} #ref={}) step:{}".format(wer_score, nhyp, nref, n_steps))
            if min_valid_wer is None or wer_score < min_valid_wer:
                min_valid_wer = wer_score
                logging.info("NEW min valid wer: {:.2f} lr={:.6f} Saving validation/model Step:{}...".format(min_valid_wer,tmos.optimizer.param_groups[0]["lr"],n_steps))
                tmos.save()
                with open("{}/validation_{}_{:.2f}.out".format(args.dir,n_steps,wer_score), 'w') as fdo:
                    fdo.write('\n'.join(generated_txts) + '\n')
                    
    wer_score, nhyp, nref, generated_txts = validation(args, tmos, valid_loader, onmttok)
    logging.info("valid wer: {:.2f} (#hyp={} #ref={}) Step:{}".format(wer_score, nhyp, nref, n_steps))
    if min_valid_wer is None or wer_score < min_valid_wer:
        min_valid_wer = wer_score
        logging.info("NEW min valid wer: {:.2f} lr={:.6f} Saving validation/model Step:{}...".format(min_valid_wer,tmos.optimizer.param_groups[0]["lr"],n_steps))
        tmos.save()
        with open("{}/validation_{}_{:.2f}.out".format(args.dir,n_steps,wer_score), 'w') as fdo:
            fdo.write('\n'.join(generated_txts) + '\n')
            

##############################################################################################################

class FormatWithEditDist():

    def __init__(self, onmttok, BAR='║', BEG='《', END='》'):
        self.onnmttok = onmttok
        self.BAR = BAR
        self.BEG = "" #BEG
        self.END = "" #END

    def __call__(self, src_txt, tgt_txt):
        src = onmttok(src_txt)
        tgt = onmttok(tgt_txt)
        sm = edit_distance.SequenceMatcher(a=src, b=tgt)
        out = []
        for opcode in sm.get_opcodes():
            code, src_beg, src_end, tgt_beg, tgt_end = opcode
            if code == 'equal':
                out.append(' '.join(src[src_beg:src_end]))
            else:
                out.append(self.BEG + ' '.join(src[src_beg:src_end]) + self.BAR + ' '.join(tgt[tgt_beg:tgt_end]) + self.END)
        return ' '.join(out)

def validation(args, tmos, loader, onmttok):
    wer_scorer = wer(onmttok)
    tmos.model.eval()
    with torch.no_grad():
        n_batchs = 0
        target_txts = []
        generated_txts = []
        #sum_loss = 0.
        for batch in loader:
            n_batchs += 1
            input_ids = batch['source_ids'].to(tmos.device, dtype = torch.long)
            attention_mask = batch['source_mask'].to(tmos.device, dtype = torch.long)
            target_ids = batch['target_ids']
            generated_ids = tmos.generate(input_ids, attention_mask, is_inference=False)
            target_txt = [tmos.decode(ids) for ids in target_ids]
            generated_txt = [tmos.decode(ids) for ids in generated_ids]
            target_txts.extend(target_txt)
            generated_txts.extend(generated_txt)
        wer_score, nhyp, nref = wer_scorer(generated_txts,target_txts)
    return wer_score, nhyp, nref, generated_txts
    
def inference(args, tmos, loader, onmttok):
    wer_scorer = wer(onmttok)
    formatWithED = FormatWithEditDist(onmttok)
    tmos.model.eval()
    with torch.no_grad():
        n_batchs = 0
        target_txts = []
        generated_txts = []
        for batch in loader:
            n_batchs += 1
            input_ids = batch['source_ids'].to(tmos.device, dtype = torch.long)
            attention_mask = batch['source_mask'].to(tmos.device, dtype = torch.long)
            generated_ids = tmos.generate(input_ids, attention_mask, is_inference=True)
            input_txt = [tmos.decode(ids)[len(args.prefix):] for ids in input_ids] ### discard initial prefix 'GEC: '
            generated_txt = [tmos.decode(ids) for ids in generated_ids] #[bsxnb, tl]           
            if 'target_ids' in batch:
                target_ids = batch['target_ids']
                target_txt = [tmos.decode(ids) for ids in target_ids]
                target_txts.extend(target_txt)

            for i in range(0, len(generated_txt), args.n_best): #if n_best is 5: generated_txt[0, 1, 2, 3, 4] are 5-bests corrections of the same input sentence
                out = []
                generated_txts.append(generated_txt[i]) ### the 1-best will be used for wer
                for pred in generated_txt[i:i+args.n_best]:
                    out.append(pred)
                if args.diffs:
                    out.append(formatWithED(input_txt[i//args.n_best], generated_txt[i]))
                    if 'target_ids' in batch:
                        out.append(formatWithED(generated_txt[i], target_txt[i]))
                        
                print('\t'.join(out))
                
        if len(target_txts):
            wer_score, nhyp, nref = wer_scorer(generated_txts,target_txts)
            logging.info("inference wer: {:.2f} (#hyp={} #ref={})".format(wer_score, nhyp, nref))

        
##############################################################################################################
##############################################################################################################
##############################################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", default=None, type=str, help="folder for model weights")
    parser.add_argument("--path", default=None, type=str, help="huggingface model path when learning from scratch (None)")
    parser.add_argument("--maxl_src", default=512, type=int, help="max length (source) sentence (512)")
    parser.add_argument("--maxl_tgt", default=128, type=int, help="max length (target) sentence (128)")
    parser.add_argument("--prefix", default="GEC:", type=str, help="prefix prepended to source sentences (GEC:)")
    parser.add_argument("--batch_sz", default=8, type=int, help="batch size (8)")
    parser.add_argument('--seed', type=int, default=23, help="seed for randomness (23)")
    parser.add_argument("--cpu", action='store_true', help="force use cpu")
    group_training = parser.add_argument_group("Training")
    group_training.add_argument("--trn_src", default=None, type=str, help="train (source) file")
    group_training.add_argument("--trn_tgt", default=None, type=str, help="train (target) file")
    group_training.add_argument("--val_src", default=None, type=str, help="valid (source) file")
    group_training.add_argument("--val_tgt", default=None, type=str, help="valid (target) file")
    group_training.add_argument("--epochs", default=1, type=int, help="number of learning epochs to run (1)")
    group_training.add_argument("--report_n", default=100, type=int, help="report every this number of steps (100)")
    group_training.add_argument("--valid_n", default=5000, type=int, help="validate every this number of steps (5000)")
    group_training.add_argument("--save_n", default=5, type=int, help="save best n checkpoints according to validation score (not implemented)")
    group_training.add_argument('--accum_n', type=int, default=1, help="accumulate this many batchs before model update (not implemented)")
    group_training.add_argument("--clip", default=0.0, type=float, help="clip to max gradient norm (0.0)")
    group_optim = parser.add_argument_group("Optimization") #AdamW
    group_optim.add_argument("--lr", default=2e-4 , type=float, help="learning rate for AdamW optimizer (2e-4)")
    group_optim.add_argument("--eps", default=1e-8, type=float, help="epsilon for AdamW optimizer (1e-8)")
    group_optim.add_argument("--beta1", default=0.9, type=float, help="beta1 for AdamW optimizer (0.9)")
    group_optim.add_argument("--beta2", default=0.999, type=float, help="beta2 for AdamW optimizer (0.999)")
    group_optim.add_argument("--wdecay", default=0, type=float, help="weight decay for AdamW optimizer (0)")
    group_optim.add_argument("--sched", default=None, type=str, help="scheduler type (None)")
    group_optim.add_argument("--warmup", default=0, type=int, help="number of warmup steps (0)")
    group_inference = parser.add_argument_group("Inference")
    group_inference.add_argument("--tst_src", default=None, type=str, help="test (source) file")
    group_inference.add_argument("--tst_tgt", default=None, type=str, help="test (target) file used for error measure")
    group_inference.add_argument("--beam_sz", default=2, type=int, help="number of beams for inference (2)")
    group_inference.add_argument("--rep_pty", default=2.5, type=float, help="repetition penalty for inference (2.5)")
    group_inference.add_argument("--len_pty", default=1.0, type=float, help="length penalty for inference (1.0)")
    group_inference.add_argument("--n_best", default=1, type=int, help="number of output sequences (1)")
    group_inference.add_argument("--early_stopping", action='store_true', help="early stopping for inference")
    group_inference.add_argument("--diffs", action='store_true', help="output src/hyp and hyp/ref diffs using edit distance")
    args = parser.parse_args()
    if args.trn_src is not None:
        logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO', None), filename=args.dir+'.log')
    else:
        logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO', None))
    if len(args.prefix) and not args.prefix.endswith(' '):
        args.prefix += ' '
    if args.beam_sz < args.n_best:
        args.beam_sz = args.n_best
    logging.info("args = {}".format(args.__dict__))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tmos = TMOS(args, device)

    ####################
    ### loading data ###
    ####################
    dsl = DataSetLoader(args, tmos.tokenizer)
    train_loader, n_train = dsl(args.trn_src, args.trn_tgt, shuffle=True) if args.trn_src is not None and args.trn_tgt is not None else (None, 0)
    valid_loader, n_valid = dsl(args.val_src, args.val_tgt, shuffle=False) if args.val_src is not None and args.val_tgt is not None else (None, 0)
    infer_loader, n_infer = dsl(args.tst_src, args.tst_tgt, shuffle=False) if args.tst_src is not None else (None, 0)
    onmttok = pyonmttok.Tokenizer("aggressive", joiner_annotate=False)

    ####################
    ### Training loop ##
    ####################
    if train_loader is not None and valid_loader is not None: 
        tmos.build_optimizer_scheduler(len(train_loader)*args.epochs)
        logging.info("Running learning...")
        tic = time.time()
        for epoch in range(1, args.epochs+1):
            train(args, epoch, tmos, train_loader, valid_loader, onmttok)
        toc = time.time()
        logging.info("learning took {:.2f} seconds, {:.2f} sentences/sec, {:.2f} batchs/sec".format(toc-tic, 100.0*n_train/(toc-tic), 100.0*len(train_loader)/(toc-tic)))

    ####################
    ### inference ######
    ####################
    if infer_loader is not None:
        logging.info("Running inference...")
        tic = time.time()
        inference(args, tmos, infer_loader, onmttok)
        toc = time.time()
        logging.info("inference took {:.2f} seconds, {:.2f} sentences/sec, {:.2f} batchs/sec".format(toc-tic, 100.0*n_infer/(toc-tic), 100.0*len(infer_loader)/(toc-tic)))
        
    logging.info("[Done]")

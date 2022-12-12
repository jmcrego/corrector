# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch
import time
import torch.optim as optim
from model.GECor import save_checkpoint
from collections import defaultdict

try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard = True
except ImportError:
    tensorboard = False


class Score():
    
    def __init__(self, tags, writer):
        self.writer = writer
        self.tags = tags
        self.tag2tp = [0] * len(tags)
        self.tag2fp = [0] * len(tags)
        self.tag2fn = [0] * len(tags)
        self.tag2n = [0] * len(tags)
        self.tag_ok = 0    #n well predicted tags
        self.tag_n = 0     #n unpadded tags
        self.tag_total = 0 #n paded/unpadded tags
        self.wrd_ok = 0
        self.wrd_n = 0
        self.wrd_total = 0
        self.Loss = 0.0
        self.nsteps = 0
        self.start = time.time()

    def addLoss(self, loss):
        self.Loss += loss #averaged per toks in batch
        self.nsteps += 1
        #logging.info('nstep={} addLoss {} => {}'.format(self.nsteps,loss,self.Loss))

    def addPred(self, outtag, outwrd, msktag, mskwrd, reftag, refwrd):
        ############
        ### tags ###
        ############
        msktag = msktag.reshape(-1) #[bs*l]
        reftag = reftag.reshape(-1) #[bs*l]
        reftag = reftag[msktag == 1]                #only unpadded
        outtag = outtag.reshape(-1,outtag.shape[2]) #[bs*l,ts]
        outtag = outtag[msktag == 1]                #only unpadded
        outtag = torch.argsort(outtag, dim=-1, descending=True)[:,0] #[bs*l, ts] => [bs*l, 1] ### get the one-best of each token        
        assert(outtag.shape == reftag.shape)
        self.tag_total += msktag.numel()
        self.tag_n += outtag.numel()
        self.tag_ok += torch.sum(reftag == outtag)

        reftag = reftag.tolist()
        outtag = outtag.tolist()
        for i in range(len(reftag)):
            self.tag2n[reftag[i]] += 1
            if reftag[i] == outtag[i]:
                self.tag2tp[reftag[i]] +=1
            else:
                self.tag2fp[outtag[i]] +=1
                self.tag2fn[reftag[i]] +=1                
        ########################
        ### word corrections ###
        ########################
        mskwrd = mskwrd.reshape(-1) #[bs*l]
        refwrd = refwrd.reshape(-1) #[bs*l]
        refwrd = refwrd[mskwrd == 1]                #only unpadded
        outwrd = outwrd.reshape(-1,outwrd.shape[2]) #[bs*l,ws]
        outwrd = outwrd[mskwrd == 1]                #only unpadded
        outwrd = torch.argsort(outwrd, dim=-1, descending=True)[:,0] #[bs*l, ws] => [bs*l, 1] ### get the one-best of each token        
        assert(outwrd.shape == refwrd.shape)
        self.wrd_total += mskwrd.numel()
        self.wrd_n += outwrd.numel()
        self.wrd_ok += torch.sum(refwrd == outwrd)

    def report(self, step, trnval):
        end = time.time()
        steps_per_sec = (self.nsteps) / (end - self.start)
        loss_per_tok = self.Loss / self.nsteps
        perc_tags_unpadded = 100.0 * self.tag_n / self.tag_total
        perc_wrds_unpadded = 100.0 * self.wrd_n / self.wrd_total
        logging.info("{}/Loss:{:.6f} step:{} steps/sec:{:.2f} unpadded tags:{:.2f}% words:{:.2f}%".format(trnval, loss_per_tok, step, steps_per_sec, perc_tags_unpadded, perc_wrds_unpadded))
        if self.writer is not None:
            self.writer.add_scalar('Loss/{}'.format(trnval), loss_per_tok, step)
            
        logging.info('{}/Acc: [tags] {}/{} ({:.1f}%) [words] {}/{} ({:.1f}%)'.format(trnval,self.tag_ok,self.tag_n,100.0*self.tag_ok/self.tag_n,self.wrd_ok,self.wrd_n,100.0*self.wrd_ok/self.wrd_n))
        if self.writer is not None:
            if self.wrd_n:
                self.writer.add_scalar('Acc/{}/cor'.format(trnval), self.wrd_ok/self.wrd_n, step)
                
        logtags = []
        for tag_idx in range(len(self.tag2n)):
            if tag_idx in [self.tags.idx_PAD, self.tags.idx_UNK]:
                continue
            tag_str = self.tags[tag_idx]
            N = self.tag2n[tag_idx]
            tp = self.tag2tp[tag_idx]
            #logging.info('{} tp={} N={}'.format(tag_str,tp,N))
            fp = self.tag2fp[tag_idx]
            fn = self.tag2fn[tag_idx]
            A = float(tp) / N if N>0 else 0.0
            P = float(tp) / (tp + fp) if tp>0 else 0.0
            R = float(tp) / (tp + fn) if tp>0 else 0.0
            F1 = 2.0*P*R / (P + R) if tp>0 else 0.0
            F2 = 5.0*P*R / (4*P + R) if tp>0 else 0.0
            logtags.append('{}|A_{:.3f}|P_{:.3f}|R_{:.3f}|F1_{:.3f}|F2_{:.3f}|N_{}'.format(tag_str,A,P,R,F1,F2,N))
            if self.writer is not None:
                self.writer.add_scalar('Acc/{}/{}'.format(trnval,tag_str), A, step)
        logging.info('{}/Tags:\t{}'.format(trnval,'\t'.join(logtags)))
            

class Learning():

    def __init__(self, model, optim, criter, step, trainset, validset, err, cor, lin, sha, args, device):
        super(Learning, self).__init__()
        writer = SummaryWriter(log_dir=args.model, comment='', purge_step=None, max_queue=10, flush_secs=60, filename_suffix='') if tensorboard else None
        n_epoch = 0
        idx_PAD_err = err.idx_PAD
        idx_PAD_cor = cor.idx_PAD if cor is not None else 2 #<pad> in FlaubertTok
        optim.zero_grad() # sets gradients to zero
        score = Score(err, writer)
        while True: #repeat epochs
            n_epoch += 1
            logging.info('Epoch {}'.format(n_epoch))
            n_batch = 0
            loss_accum = 0.0
            dsrc = {}
            for src, agg, err, cor, _, _ in trainset:
                bs, l = src.shape
                src = src.to(device) #[bs,l]
                agg = agg.to(device) #[bs,l]
                err = err.to(device) #[bs,l]
                cor = cor.to(device) #[bs,l,n_subtokens]
                model.train()
                criter.train()
                dsrc['input_ids'] = src
                outerr, outcor = model(dsrc, agg) #[bs, l, v_err], [bs, l, v_cor*n_subtokens] (forward, no log_softmax is applied)
                assert(outerr.shape[0] == err.shape[0] and outerr.shape[1] == err.shape[1])
                #logging.info('outerr.shape={} err.shape={}'.format(outerr.shape,err.shape))
                if args.n_subtokens > 1:
                    cor = cor.reshape([bs,l*args.n_subtokens])
                    outcor = outcor.reshape([bs,l*args.n_subtokens,-1])
                assert(outcor.shape[0] == cor.shape[0] and outcor.shape[1] == cor.shape[1])                
                #logging.info('outcor.shape={} cor.shape={}'.format(outcor.shape,cor.shape))
                mskerr = torch.ones_like(err).to(device)
                mskcor = torch.ones_like(cor).to(device)
                mskerr[err == idx_PAD_err] = 0
                mskcor[cor == idx_PAD_cor] = 0
                if mskerr.sum() == 0 or mskcor.sum() == 0:
                    #logging.warning('Discarded train batch: all tokens padded')
                    continue
                n_batch += 1
                loss = criter(outerr, outcor, mskerr, mskcor, err, cor) / float(args.accum_n_batchs) #average of losses in batch (already normalized by tokens in batch) (n batchs will be accumulated before model update, so i normalize by n batchs)
                score.addPred(outerr, outcor, mskerr, mskcor, err, cor)
                loss.backward()
                loss_accum += loss
                #logging.info('loss {:.6f} loss_acum {:.6f}'.format(loss,loss_accum))
                if n_batch % args.accum_n_batchs == 0:
                    step += 1 ### current step
                    ### optimize ###
                    score.addLoss(loss_accum.item())
                    if args.clip > 0.0: # clip gradients norm
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optim.step() # updates model parameters
                    optim.zero_grad() # sets gradients to zero for next update
                    loss_accum = 0.0
                    ### report ###
                    if args.report_every and step % args.report_every == 0:
                        score.report(step, 'train')
                        score = Score(errs, writer)
                    ### save ###
                    if args.save_every and step % args.save_every == 0: 
                        save_checkpoint(args.model, model, optim, step, args.keep_last_n)
                    ### validate ###
                    if args.validate_every and step % args.validate_every == 0: 
                        self.validate(model, criter, step, validset, tags, idx_PAD_err, idx_PAD_cor, args, writer, device)
                    ### stop by max_steps ###
                    if args.max_steps and step >= args.max_steps: 
                        self.validate(model, criter, step, validset, tags, idx_PAD_err, idx_PAD_cor, args, writer, device)
                        save_checkpoint(args.model, model, optim, step, args.keep_last_n)
                        logging.info('Learning STOP by [steps={}]'.format(step))
                        return
            ### stop by max_epochs ###
            if args.max_epochs and n_epoch >= args.max_epochs:
                self.validate(model, criter, step, validset, tags, idx_PAD_err, idx_PAD_cor, args, writer, device)
                save_checkpoint(args.model, model, optim, step, args.keep_last_n)
                logging.info('Learning STOP by [epochs={}]'.format(n_epoch))
                return

    def validate(self, model, criter, step, validset, tags, idx_PAD_err, idx_PAD_cor, args, writer, device):
        if validset is None:
            return
        model.eval()
        criter.eval()
        score = Score(tags, writer)
        with torch.no_grad():
            dsrc = {}
            for src, agg, err, cor, _, _ in validset:
                src = src.to(device)
                agg = agg.to(device)
                err = err.to(device)
                cor = cor.to(device)
                dsrc['input_ids'] = src
                outerr, outcor = model(dsrc, agg) ### forward
                if args.n_subtokens > 1:
                    bs, l, cs = outcor.shape
                    cor = cor.reshape([bs,l*args.n_subtokens])
                    outcor = outcor.reshape([bs,l*args.n_subtokens,-1])
                
                mskerr = torch.ones_like(err).to(err.device)
                mskcor = torch.ones_like(cor).to(cor.device)
                mskerr[err == idx_PAD_err] = 0
                mskcor[cor == idx_PAD_cor] = 0
                if mskerr.sum() == 0 or mskcor.sum() == 0:
                    logging.warning('Discarded valid batch: all tokens padded')
                    continue
                loss = criter(outerr, outcor, mskerr, mskcor, err, cor)
                score.addLoss(loss.item())
                score.addPred(outerr, outcor, mskerr, mskcor, err, cor)
        score.report(step,'valid')


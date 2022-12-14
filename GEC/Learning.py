# -*- coding: utf-8 -*-
import sys
import os
import logging
import numpy as np
import torch
import time
import torch.optim as optim
from GEC.GECor import save_checkpoint
from collections import defaultdict

try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard = True
except ImportError:
    tensorboard = False


class Score():
    
    def __init__(self, tags, writer):
        self.writer = writer
        self.tags = tags #vocabulary of tags
        self.tag2tp = [0] * len(tags)
        self.tag2fp = [0] * len(tags)
        self.tag2fn = [0] * len(tags)
        self.tag2n = [0] * len(tags)
        self.tag_ok = 0    #n well predicted tags
        self.tag_n = 0     #n unpadded tags
        self.tag_total = 0 #n paded/unpadded tags
        self.stok_ok = 0
        self.stok_n = 0
        self.stok_total = 0
        self.Loss = 0.0
        self.nsteps = 0
        self.start = time.time()

    def addLoss(self, loss):
        self.Loss += loss #averaged per toks in batch
        self.nsteps += 1
        #logging.info('nstep={} addLoss {} => {}'.format(self.nsteps,loss,self.Loss))

    def addPred(self, outtag, outwrd, msktag, mskwrd, reftag, refwrd):
        bs, l, vtag = outtag.shape
        bs, l_x_n_subtoks, vcor = outwrd.shape
        #n_subtoks = l_x_n_subtoks / l
        #
        #outtag.shape torch.Size([81, 50, 15])
        #msktag.shape torch.Size([81, 50])
        #reftag.shape torch.Size([81, 50])
        #
        #outwrd.shape torch.Size([81, 500, 32128])
        #mskwrd.shape torch.Size([81, 500])
        #refwrd.shape torch.Size([81, 500])
        ############
        ### tags ###
        ############
#        for i in range(bs):
#            logging.info('msktag[{}]: {}'.format(i,msktag[i].tolist()))
#            logging.info('reftag[{}]: {}'.format(i,reftag[i].tolist()))
        
        msktag = msktag.reshape(-1) #[bs*l]
        reftag = reftag.reshape(-1) #[bs*l]
        reftag = reftag[msktag == 1] #only unpadded [bs*?]
        outtag = outtag.reshape(-1,outtag.shape[2]) #[bs*l,vtag]
        outtag = outtag[msktag == 1] #only unpadded [bs*?, vtag]
        outtag = torch.argsort(outtag, dim=-1, descending=True)[:,0] #[bs*?, vtag] => [bs*?, 1] => [bs*?] ### get the one-best of each token (tags predicted)
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
        ############
        ### cors ###
        ############
        mskwrd = mskwrd.reshape(-1) #[bs*l*n_subtoks]
        refwrd = refwrd.reshape(-1) #[bs*l*n_subtoks]
        outwrd = outwrd.reshape(-1,vcor) #[bs*l*n_subtoks,vcor]
        refwrd = refwrd[mskwrd == 1] #only unpadded #[bs*?*n_subtoks]
        outwrd = outwrd[mskwrd == 1] #only unpadded [bs*?*n_subtoks,vcor]
        outwrd = torch.argsort(outwrd, dim=-1, descending=True)[:,0] #[bs*?*n_subtoks, vcor] => [bs*?*n_subtoks, 1] => [bs*?*n_subtoks] ### get the one-best of each token        
        assert(outwrd.shape == refwrd.shape)
        self.stok_total += mskwrd.numel()
        self.stok_n += outwrd.numel()
        self.stok_ok += torch.sum(refwrd == outwrd)

    def report(self, step, trnval):
        end = time.time()
        steps_per_sec = (self.nsteps) / (end - self.start)
        loss_per_tok = self.Loss / self.nsteps
        perc_tags_unpadded = 100.0 * self.tag_n / self.tag_total
        perc_wrds_unpadded = 100.0 * self.stok_n / self.stok_total
        logging.info("{}/Loss:{:.6f} step:{} steps/sec:{:.2f} unpadded tags:{:.2f}% unpadded subtoks:{:.2f}%".format(trnval, loss_per_tok, step, steps_per_sec, perc_tags_unpadded, perc_wrds_unpadded))
        if self.writer is not None:
            self.writer.add_scalar('Loss/{}'.format(trnval), loss_per_tok, step)
            
        logging.info('{}/Acc: [tags] {}/{} ({:.1f}%) [stoks] {}/{} ({:.1f}%)'.format(trnval,self.tag_ok,self.tag_n,100.0*self.tag_ok/self.tag_n,self.stok_ok,self.stok_n,100.0*self.stok_ok/self.stok_n))
        if self.writer is not None:
            if self.stok_n:
                self.writer.add_scalar('Acc/{}/cor'.format(trnval), self.stok_ok/self.stok_n, step)
                
        for tag_idx in range(len(self.tag2n)):
            if tag_idx in [self.tags.idx_PAD, self.tags.idx_UNK]:
                continue
            tag_str = self.tags[tag_idx]
            N = self.tag2n[tag_idx]
            tp = self.tag2tp[tag_idx]
            fp = self.tag2fp[tag_idx]
            fn = self.tag2fn[tag_idx]
            A = float(tp) / N if N>0 else 0.0
            P = float(tp) / (tp + fp) if tp>0 else 0.0
            R = float(tp) / (tp + fn) if tp>0 else 0.0
            F = 2.0*P*R / (P + R) if tp>0 else 0.0
            F2 = 5.0*P*R / (4*P + R) if tp>0 else 0.0
            logging.info("{}/{}\tA:{:.3f} P:{:.3f} R:{:.3f} F:{:.3f} N:{}".format(trnval,tag_str,A,P,R,F,N))
            if self.writer is not None:
                self.writer.add_scalar('Acc/{}/{}'.format(trnval,tag_str), A, step)
            

class Learning():

    def __init__(self, model, optim, criter, step, trainset, validset, tags, idx_PAD_cor, args, device):
        super(Learning, self).__init__()
        writer = SummaryWriter(log_dir=args.model, comment='', purge_step=None, max_queue=10, flush_secs=60, filename_suffix='') if tensorboard else None
        n_epoch = 0
        idx_PAD_err = tags.idx_PAD #<pad> in err vocab
        optim.zero_grad() # sets gradients to zero
        score = Score(tags, writer)
        while True: #repeat epochs
            n_epoch += 1
            logging.info('Epoch {}'.format(n_epoch))
            n_batch = 0
            loss_accum = 0.0
            for src, agg, err, cor, _ in trainset:
                src = src.to(device) #[bs,l1]
                agg = agg.to(device) #[bs,l1]
                err = err.to(device) #[bs,l2]
                cor = cor.to(device) #[bs,l2,n_subtok]
                assert src.shape[0] == agg.shape[0] == err.shape[0] == cor.shape[0]
                assert src.shape[1] == agg.shape[1]
                assert err.shape[1] == cor.shape[1]
                assert cor.shape[2] == args.n_subtok
                bs, l1 = src.shape
                bs, l2 = err.shape
                model.train()
                criter.train()
                outerr, outcor = model(src, agg, l2=l2) #[bs, l2, v_err], [bs, l2, n_subtok*v_cor] (forward, no log_softmax is applied)
                outerr = outerr.reshape([bs,l2,-1])
                outcor = outcor.reshape([bs,l2,-1])
                outcor = outcor.reshape([bs,l2*args.n_subtok,-1]) #[bs,l2*n_subtok, v_cor]
                cor = cor.reshape([bs,l2*args.n_subtok])
                mskerr = torch.ones_like(err).to(device) #[bs, l2]
                mskcor = torch.ones_like(cor).to(device) #[bs, l2*n_subtok, v_cor]
                mskerr[err == idx_PAD_err] = 0 #masked set to 0 rest to 1
                mskcor[cor == idx_PAD_cor] = 0 #masked set to 0 rest to 1
                if mskerr.sum() == 0 or mskcor.sum() == 0:
                    continue #Discarded train batch (all tokens padded)
                n_batch += 1
                loss = criter(outerr, outcor, mskerr, mskcor, err, cor) / float(args.accum_n) #average of losses in batch (already normalized by tokens in batch) (n batchs will be accumulated before model update, so i normalize by n batchs)
                score.addPred(outerr, outcor, mskerr, mskcor, err, cor)
                loss.backward()
                loss_accum += loss
                if n_batch % args.accum_n == 0:
                    step += 1 ### current step
                    ### optimize ###
                    score.addLoss(loss_accum.item())
                    if args.clip > 0.0: # clip gradients norm
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optim.step() # updates model parameters
                    optim.zero_grad() # sets gradients to zero for next update
                    loss_accum = 0.0
                    ### report ###
                    if args.report_n and step % args.report_n == 0:
                        score.report(step, 'train')
                        score = Score(tags, writer)
                    ### save ###
                    if args.save_n and step % args.save_n == 0: 
                        save_checkpoint(args.model, model, optim, step, args.keep_n)
                    ### validate ###
                    if args.valid and args.valid_n and step % args.valid_n == 0: 
                        self.validate(model, criter, step, validset, tags, idx_PAD_err, idx_PAD_cor, args, writer, device)
                    ### stop by max_steps ###
                    if args.steps and step >= args.steps:
                        if args.valid:
                            self.validate(model, criter, step, validset, tags, idx_PAD_err, idx_PAD_cor, args, writer, device)
                        save_checkpoint(args.model, model, optim, step, args.keep_n)
                        logging.info('Learning STOP by [steps={}]'.format(step))
                        return
            ### stop by max_epochs ###
            if args.epochs and n_epoch >= args.epochs:
                if args.valid:
                    self.validate(model, criter, step, validset, tags, idx_PAD_err, idx_PAD_cor, args, writer, device)
                save_checkpoint(args.model, model, optim, step, args.keep_n)
                logging.info('Learning STOP by [epochs={}]'.format(n_epoch))
                return

    def validate(self, model, criter, step, validset, tags, idx_PAD_err, idx_PAD_cor, args, writer, device):
        if validset is None:
            return
        model.eval()
        criter.eval()
        score = Score(tags, writer)
        with torch.no_grad():
            for src, agg, err, cor, _, in validset:
                src = src.to(device)
                agg = agg.to(device)
                err = err.to(device)
                cor = cor.to(device)
                outerr, outcor = model(src, agg) ### forward
                if args.n_subtok > 1:
                    bs, l, cs = outcor.shape
                    cor = cor.reshape([bs,l*args.n_subtok])
                    outcor = outcor.reshape([bs,l*args.n_subtok,-1])
                
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


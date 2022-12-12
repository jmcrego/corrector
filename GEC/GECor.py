import os
import sys
import glob
import torch
import logging
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
import itertools

def load_or_create_checkpoint(fmodel, model, optimizer, device):
    files = sorted(glob.glob("{}.????????.pt".format(fmodel)))
    if len(files) == 0:
        step = 0
        save_checkpoint(fmodel, model, optimizer, step, 0)
    else:
        step, model, optimizer = load_checkpoint(fmodel, model, optimizer, device)
    return step, model, optimizer


def load_checkpoint(fmodel, model, optimizer, device):
    step = 0
    files = sorted(glob.glob("{}.????????.pt".format(fmodel)))
    if len(files) == 0:
        logging.info('No model found')
        sys.exit()
    file = files[-1] ### last is the newest
    checkpoint = torch.load(file, map_location=device)
    step = checkpoint['step']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logging.info('Loaded checkpoint step={} from {} device={}'.format(step,fmodel,device))
    return step, model, optimizer

def load_model(fmodel, model, device):
    checkpoint = torch.load(fmodel, map_location=device)
    model.load_state_dict(checkpoint['model'])
    logging.info('Loaded checkpoint from {} device={}'.format(fmodel,device))
    return model
    
def save_checkpoint(fmodel, model, optimizer, step, keep_last_n):
    if os.path.isfile("{}.{:08d}.pt".format(fmodel,step)):
        logging.info('Checkpoint already exists')
        return
    checkpoint = { 'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict() }
    torch.save(checkpoint, "{}.{:08d}.pt".format(fmodel,step))
    logging.info('Saved checkpoint step={} in {}.{:08d}.pt'.format(step,fmodel,step))
    files = sorted(glob.glob(fmodel + '.????????.pt')) 
    while keep_last_n > 0 and len(files) > keep_last_n:
        f = files.pop(0)
        os.remove(f) ### first is the oldest
        logging.debug('Removed checkpoint {}'.format(f))

    
class CE2(torch.nn.Module):

    def __init__(self, label_smoothing=0.0, beta=1.0):
        super(CE2, self).__init__()
        self.crossent = nn.CrossEntropyLoss(label_smoothing=label_smoothing,reduction='mean') #only tokens not padded are used to compute loss
        self.beta = beta
        logging.info('Built CE2 ls={} beta={}'.format(label_smoothing, beta))

    def forward(self, outerr, outcor, mskerr, mskcor, referr, refcor):
        #loss_err = self.crossent(outerr[mskerr.bool()], referr[mskerr.bool()])
        #loss_cor = self.crossent(outcor[mskcor.bool()], refcor[mskcor.bool()])
        #return loss_err + self.beta * loss_cor
        (bs, lt, ts) = outerr.shape
        (_,  lc, cs) = outcor.shape
        #logging.info('outerr.shape = {}'.format(outerr.shape)) #[bs, lt, ts]
        #logging.info('outcor.shape = {}'.format(outcor.shape)) #[bs, lc, cs]
        #logging.info('mskerr.shape = {}'.format(mskerr.shape)) #[bs, lt]
        #logging.info('mskcor.shape = {}'.format(mskcor.shape)) #[bs, lc, 1]
        #logging.info('referr.shape = {}'.format(referr.shape)) #[bs, lt]
        #logging.info('refcor.shape = {}'.format(refcor.shape)) #[bs, lc, 1]
        outerr = outerr.reshape(bs*lt,-1) #[bs*lt,ts]
        outcor = outcor.reshape(bs*lc,-1) #[bs*lc,cs]
        mskerr = mskerr.reshape(bs*lt) #[bs*lt]
        mskcor = mskcor.reshape(bs*lc) #[bs*lc]
        referr = referr.reshape(bs*lt) #[bs*lt]
        refcor = refcor.reshape(bs*lc) #[bs*lc]

        outerr_mask = outerr[mskerr.bool()] #[N,ts]
        referr_mask = referr[mskerr.bool()] #[N]
        loss_err = self.crossent(outerr_mask, referr_mask)
        #logging.info('loss_err={} : {} out of {} elements'.format(loss_err.item(), mskerr.sum(), torch.numel(mskerr)))

        outcor_mask = outcor[mskcor.bool()] #[M,cs]
        refcor_mask = refcor[mskcor.bool()] #[M]
        loss_cor = self.crossent(outcor_mask, refcor_mask) 
        #logging.info('loss_cor={} : {} out of {} elements'.format(loss_cor.item(), mskcor.sum(), torch.numel(mskcor)))

        loss = loss_err + self.beta * loss_cor
        logging.debug('CE2 loss: {:.3f} + {:.3f} = {:.3f}'.format(loss_err.item(),loss_cor.item(),loss.item()))
        return loss    

    
class GECor(nn.Module):

    def __init__(self, model, err, n_subtokens, merge='sum'):
        super(GECor, self).__init__() 
        self.model = model #for model details: print(model.config)
        self.idx_PAD_err = err.idx_PAD
        self.n_err = len(err)
        self.n_cor = n_subtokens * self.model.config.vocab_size
        self.merge = merge
        self.n_subtokens = n_subtokens
        self.emb_size = self.model.config.d_model
        self.linear_layer_err = nn.Linear(self.emb_size, self.n_err)
        self.linear_layer_cor = nn.Linear(self.emb_size, self.n_cor)
        
    def forward(self, inputs, indexs):
        ### encoder layer ###
        embeddings = self.model.get_input_embeddings()(inputs) #[bs, l, ed]
        if torch.max(indexs) > embeddings.shape[1]-1:
            logging.error('Indexs bad formatted!')
            sys.exit()
        ### merge into words ###
        if self.merge in ['sum', 'avg', 'max']:
            embeddings_merged = torch.zeros_like(embeddings, dtype=embeddings.dtype, device=embeddings.device)
            torch_scatter.segment_coo(embeddings, indexs, out=embeddings_merged, reduce=self.merge)
        elif self.merge in ['first','last']:
            embeddings_merged = embeddings #not finished!!!
        else:
            logging.error('Bad merge value: {}'.format(self.merge))
            sys.exit()
        ### out layers ###
        out_err = self.linear_layer_err(embeddings_merged) #[bs, l, es]
        out_cor = self.linear_layer_cor(embeddings_merged) #[bs, l, cs]        
        return out_err, out_cor

    def parameters(self):
        return super().parameters()    
    
    

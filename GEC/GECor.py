import os
import sys
import glob
import torch
import logging
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from transformers import FlaubertModel
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

    def __init__(self, err, cor, lin, sha, encoder_name="flaubert/flaubert_base_cased", aggregation='sum',shapes_size=0, n_subtokens=0):
        super(GECor, self).__init__() #flaubert_base_cased info in https://huggingface.co/flaubert/flaubert_base_cased/tree/main can be accessed via self.encoder.config.vocab_size

        self.encoder = FlaubertModel.from_pretrained(encoder_name) #Flaubert Encoder
        self.shapes_encoder = nn.Embedding(len(sha), shapes_size) if shapes_size > 0 and sha is not None else None #Encoder of additional input (shapes, inlex)
        
        self.idx_PAD_err = err.idx_PAD
        self.idx_PAD_cor = cor.idx_PAD if cor is not None else None
        self.idx_PAD_lin = lin.idx_PAD if lin is not None else None
        self.idx_PAD_COR = 2 #"<pad>": 2, https://huggingface.co/flaubert/flaubert_base_cased/raw/main/vocab.json

        self.n_err = len(err)
        self.n_cor = len(cor) if cor is not None else None
        self.n_lin = len(lin) if lin is not None else None
        self.n_COR = n_subtokens * self.encoder.config.vocab_size if n_subtokens > 0 else None #n_subtokens * 68729 #https://huggingface.co/flaubert/flaubert_base_cased/tree/main can be accessed via self.encoder.config.vocab_size
        
        self.aggregation = aggregation
        self.n_subtokens = n_subtokens
        self.emb_size = self.encoder.config.emb_dim
        if shapes_size > 0:
            self.emb_size += shapes_size
        self.linear_layer_err = nn.Linear(self.emb_size, self.n_err)
        self.linear_layer_cor = nn.Linear(self.emb_size, self.n_cor) if self.n_cor is not None else None
        self.linear_layer_lin = nn.Linear(self.emb_size, self.n_lin) if self.n_lin is not None else None
        self.linear_layer_COR = nn.Linear(self.emb_size, self.n_COR) if self.n_COR is not None else None
        
    def forward(self, inputs, indexs, shapes_inputs=None):
        #####################
        ### encoder layer ###
        #####################
        embeddings = self.encoder(**inputs).last_hidden_state #[bs, l, ed]
        if torch.max(indexs) > embeddings.shape[1]-1:
            logging.error('Indexs bad formatted!')
            sys.exit()

        ###################
        ### aggregation ###
        ###################
        if self.aggregation in ['sum', 'avg', 'max']:
            embeddings_aggregate = torch.zeros_like(embeddings, dtype=embeddings.dtype, device=embeddings.device)
            torch_scatter.segment_coo(embeddings, indexs, out=embeddings_aggregate, reduce=self.aggregation)
        elif self.aggregation in ['first','last']:
            embeddings_aggregate = embeddings #not finished!!!
        else:
            logging.error('Bad aggregation value: {}'.format(self.aggregation))
            sys.exit()

        ##################################################
        ### embeddings_aggregate + shapes_encoder(shapes_inputs) ###
        ##################################################
        if shapes_inputs is not None and self.shapes_encoder is not None:
            embeddings_shapes =  self.shapes_encoder(shapes_inputs) #[bs, l, eS]
            embeddings_aggregate = torch.cat((embeddins_aggregate,embeddings_shapes), -1) #[bs, l, es+eS]
            
        ##################
        ### out layers ###
        ##################
        out_err = self.linear_layer_err(embeddings_aggregate) #[bs, l, es]
        out_cor = self.linear_layer_cor(embeddings_aggregate) if self.linear_layer_cor is not None else None #[bs, l, cs]
        out_lin = self.linear_layer_lin(embeddings_aggregate) if self.linear_layer_lin is not None else None #[bs, l, ls]
        out_COR = self.linear_layer_COR(embeddings_aggregate) if self.linear_layer_COR is not None else None #[bs, l, Cs]
        
        return out_err, out_cor, out_lin, out_COR

    def parameters(self):
        return super().parameters()    
    
    

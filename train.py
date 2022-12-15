import sys
import os
import time
import random
import logging
import torch
import argparse
import torch.optim as optim
from transformers import T5EncoderModel, T5Tokenizer
from GEC.GECor import GECor, load_or_create_checkpoint, load_checkpoint, save_checkpoint, CE2
from GEC.Learning import Learning
from GEC.Dataset import Dataset
from GEC.Vocab import Vocab

######################################################################
### MAIN #############################################################
######################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Model file')
    
    parser.add_argument('--train', help='Training file (required)', required=True)
    parser.add_argument('--valid', help='Validation file')
    parser.add_argument('--cuda', action='store_true', help='Use cuda device instead of cpu')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomness (0)')
    
    group_network = parser.add_argument_group("Network")
    group_network.add_argument('--err_voc', type=str, default=None, help='Error vocabulary (required)', required=True)
    group_network.add_argument('--n_subtok', type=int, default=10,    help='Number of correction subtokens (10)')
    group_network.add_argument('--merge', type=str, default="max",help='Merge subtokens: first, last, max, avg, sum (max)')
    
    group_optim = parser.add_argument_group("Optim")
    group_optim.add_argument('--lr', type=float, default=0.00001, help='Learning Rate (0.00001)')
    group_optim.add_argument('--beta', type=float, default=1.0, help='Beta for CE2 loss, l = l_err + β * l_cor (1.0)')
    group_optim.add_argument('--clip', type=float, default=0.0, help='Clip gradient norm of parameters (0.0)')
    group_optim.add_argument('--ls', type=float, default=0.1, help='Label smoothing value (0.1)')
    group_optim.add_argument('--batch_sz', type=int, default=4096, help='Batch size (4096)')
    group_optim.add_argument('--batch_tp', type=str, default="tokens", help='Batch type: tokens or sentences (tokens)')
    group_optim.add_argument('--accum_n', type=int, default=4, help='Accumulate n batchs before model update (4)')
    group_optim.add_argument('--freeze_enc', action='store_true', help='Freeze encoder parameters')
    group_optim.add_argument('--freeze_err', action='store_true', help='Freeze err linear layer parameters')
    group_optim.add_argument('--freeze_cor', action='store_true', help='Freeze cor linear layer parameters')

    group_learning = parser.add_argument_group("Learning")
    group_learning.add_argument('--shard_sz', type=int, default=2000000, help='Examples in shard (2000000)')
    group_learning.add_argument('--max_len', type=int, default=200, help='Maximum length in words (200)')
    group_learning.add_argument('--steps', type=int, default=0, help='Maximum training steps (0)')
    group_learning.add_argument('--epochs', type=int, default=0, help='Maximum training epochs (0)')
    group_learning.add_argument('--valid_n', type=int, default=5000, help='Validate every this steps (5000)')
    group_learning.add_argument('--save_n', type=int, default=10000, help='Save model every this steps (10000)')
    group_learning.add_argument('--report_n', type=int, default=100, help='Report every this steps (100)')
    group_learning.add_argument('--keep_n', type=int, default=2, help='Save last n models (2)')
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'DEBUG' if args.debug else 'INFO', None), filename=args.model+'.log')

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    logging.info("Options = {}".format(args.__dict__))
    
    tic = time.time()    
    ########################
    ### load model/optim ###
    ########################
    path = 't5-small'
    t5mod = T5EncoderModel.from_pretrained(path)
    t5tok = T5Tokenizer.from_pretrained(path, model_max_length=t5mod.config.n_positions)
    err = Vocab(args.err_voc)
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    model = GECor(t5mod, err, args.n_subtok, args.merge).to(device)
    optim = optim.Adam(model.parameters(), lr=args.lr)
    last_step, model, optim = load_or_create_checkpoint(args.model, model, optim, device)
    criter = CE2(args.ls,args.beta).to(device)
    model.freeze_enc()
    sys.exit()
    
    #############
    ### learn ###
    #############
    validset = Dataset(args.valid, err, t5tok, args) if args.valid else None
    trainset = Dataset(args.train, err, t5tok, args)
    learning = Learning(model, optim, criter, last_step, trainset, validset, err, t5tok.pad_token_id, args, device)
    
    toc = time.time()
    logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    

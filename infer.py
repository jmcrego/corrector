import sys
import os
import time
import random
import logging
import torch
import argparse
import torch.optim as optim
from GEC.GECor import GECor, load_or_create_checkpoint, load_checkpoint, save_checkpoint, CE2, load_model
from GEC.Inference import Inference
from GEC.Dataset import Dataset
from GEC.Vocab import Vocab
#from Lexicon import Lexicon
#from FlaubertTok import FlaubertTok
#from Tokenizer import Tokenizer
#from Analyzer import Analyzer
#from Utils import create_logger

######################################################################
### MAIN #############################################################
######################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Model file')
    
    parser.add_argument('--test', help='Testing data file (required)', required=True)
    parser.add_argument('--cuda', action='store_true', help='Use cuda device instead of cpu')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomness (0)')
    ### network
    group_network = parser.add_argument_group("Network (must be the values used in training)")
    group_network.add_argument('--err_voc', type=str, default=None, help='Error vocabulary (required)', required=True)
    group_network.add_argument('--n_subtok', type=int, default=10,    help='Number of correction subtokens (10)')
    group_network.add_argument('--merge', type=str, default="max",help='Merge subtokens: first, last, max, avg, sum (max)')
    ### inference
    group_infer = parser.add_argument_group("Inference")
    group_infer.add_argument('--n_best', type=int, default=10, help='Output N-best error tag options (10)')    
    group_infer.add_argument('--max_iter', type=int, default=3, help='Max number of correction iterations over one sentence (3)')    
    group_infer.add_argument('--max_len', type=int, default=0, help='Maximum example length (0)')    
    group_infer.add_argument('--shard_sz', type=int, default=5000000, help='Examples in shard (5000000)')
    group_infer.add_argument('--batch_sz', type=int, default=4096, help='Batch size (4096)')
    group_infer.add_argument('--batch_tp', type=str, default="tokens", help='Batch type: tokens or sentences (tokens)')
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'DEBUG' if args.debug else 'INFO', None))

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    logging.info("Options = {}".format(args.__dict__))

    ########################
    ### load model/optim ###
    ########################
    path = 't5-small'
    t5tok = T5Tokenizer.from_pretrained(path, model_max_length=t5mod.config.n_positions)
    err = Vocab(args.err_voc)
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    model = GECor(t5mod, err, args.n_subtok, args.merge).to(device)
    model = load_model(args.model,model,device)

    #############
    ### infer ###
    #############
    tic = time.time()
    testset = Dataset(args.test, err, t5tok, args)
    inference = Inference(model, testset, tags, cors, token, lex, args, device)
    toc = time.time()
    logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    

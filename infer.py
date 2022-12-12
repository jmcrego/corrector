import sys
import os
import time
import random
import logging
import torch
import argparse
import torch.optim as optim
from GECor import GECor, load_or_create_checkpoint, load_checkpoint, save_checkpoint, CE2, load_model
from Inference import Inference
from Dataset import Dataset
from Vocab import Vocab
from Lexicon import Lexicon
from FlaubertTok import FlaubertTok
#from Tokenizer import Tokenizer
#from Analyzer import Analyzer
from Utils import create_logger

######################################################################
### MAIN #############################################################
######################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Model file (required)', required=True)
    parser.add_argument('--test', help='Testing data file (required)', required=True)
    parser.add_argument('--lex', help='Lexicon file (required)', required=True)
    ### network
    parser.add_argument('--tags', help='Vocabulary of tags (required)', required=True)
    parser.add_argument('--cors', help='Vocabulary of corrections', required=False)
    parser.add_argument('--n_subtokens', type=int, default=1, help='Number of word subtokens to predict', required=False)
    parser.add_argument('--aggregation', type=str, default="max", help='Aggregation when merging embeddings: first, last, max, avg, sum (max)')
    ### inference
    parser.add_argument('--Kt', type=int, default=10, help='Output K-best tag options (10)')    
    parser.add_argument('--Kc', type=int, default=10, help='Output K-best word options (10)')    
    parser.add_argument('--max_iter', type=int, default=3, help='Max number of correction iterations over one sentence (3)')    
    ### data
    parser.add_argument('--shard_size', type=int, default=5000000, help='Examples in shard (5000000)')
    parser.add_argument('--max_length', type=int, default=0, help='Maximum example length (0)')    
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size (4096)')
    parser.add_argument('--batch_type', type=str, default="tokens", help='Batch type: tokens or sentences (tokens)')
    ### others
    parser.add_argument('--cuda', action='store_true', help='Use cuda device instead of cpu')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomness (0)')
    parser.add_argument('--log_file', type=str, default="stderr", help='Log file (stderr)')
    parser.add_argument('--log', type=str, default="info", help='Log level (info)')
    args = parser.parse_args()

    create_logger(args.log_file,args.log)
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    logging.info("Options = {}".format(args.__dict__))

    token = FlaubertTok("flaubert/flaubert_base_cased")
    tags = Vocab(args.tags)
    cors = Vocab(args.cors) if args.cors is not None else None
    lex = Lexicon(args.lex)

    #testset = Dataset(args.test, tags, cors, token, args)
    #for i in range(len(testset.Data)):
    #    print("{}\t{}".format(testset.Data[i]['idx'],testset.Data[i]['raw']))
    #    print("\tids_src\t{}".format(testset.Data[i]['ids_src']))
    #    print("\tstr_src\t{}".format(testset.Data[i]['str_src']))
    #sys.exit()
    
    ########################
    ### load model/optim ###
    ########################
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    model = GECor(tags, cors, encoder_name="flaubert/flaubert_base_cased", aggregation=args.aggregation, n_subtokens=args.n_subtokens).to(device)
    model = load_model(args.model,model,device)

    #############
    ### infer ###
    #############
    tic = time.time()
    testset = Dataset(args.test, tags, cors, token, args)
    inference = Inference(model, testset, tags, cors, token, lex, args, device)
    toc = time.time()
    logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    

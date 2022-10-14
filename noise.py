import re
import sys
import time
import random
import logging
import argparse
import pyonmttok
from collections import defaultdict
from Noises import Misspell, Case, Hyphen, Space, Duplicate, Replacement, ONMTTOK_JOINER

def del_joiners(txt):
    txt = txt[:]
    starts_with_joiner, ends_with_joiner = False, False
    if txt.startswith(ONMTTOK_JOINER):
        txt = txt[1:]
        starts_with_joiner = True
    if txt.endswith(ONMTTOK_JOINER):
        txt = txt[:-1]
        ends_with_joiner = True
    return txt, starts_with_joiner, ends_with_joiner
        
def add_joiners(txt, starts_with_joiner, ends_with_joiner):
    if starts_with_joiner:
        txt = ONMTTOK_JOINER + txt
    if ends_with_joiner:
        txt = txt + ONMTTOK_JOINER
    return txt

def get_error_order(args, misspell, grammar, homophone, case, space, duplicate):
    ### set the order of the sequence of errors according to the given weights
    errors2weights = {}
    if args.w_misspell and misspell is not None:
        errors2weights['misspell'] = args.w_misspell
    if args.w_homophone and homophone is not None:
        errors2weights['homophone'] = args.w_homophone
    if args.w_grammar and grammar is not None:
        errors2weights['grammar'] = args.w_grammar
    if args.w_case and case is not None:
        errors2weights['case'] = args.w_case
    if args.w_hyphen:
        errors2weights['hyphen'] = args.w_hyphen
    if args.w_space:
        errors2weights['space'] = args.w_space
    if args.w_duplicate:
        errors2weights['duplicate'] = args.w_duplicate
    next_errors = []
    for _ in range(len(errors2weights)):
        error = random.choices(list(errors2weights.keys()), list(errors2weights.values()), k=1)[0]
        next_errors.append(error)
        del errors2weights[error]
    return next_errors
        
def noise_token(txt_clean, prev_tok, post_tok, args, stats, misspell, grammar, homophone, case, hyphen, space, duplicate):
    txt, starts_with_joiner, ends_with_joiner = del_joiners(txt_clean)
    if txt.isnumeric():
        return txt_clean, None
    
    next_errors = get_error_order(args, misspell, grammar, homophone, case, space, duplicate)
    ### try all errors in next_errors until one is injected
    for next_error in next_errors:
        
        if next_error == 'grammar':
            txt_noised = grammar(str(txt))
            if txt_noised is not None and txt_noised != txt:
                return add_joiners(txt_noised, starts_with_joiner, ends_with_joiner), 'grammar'
        
        elif next_error == 'homophone':
            txt_noised = homophone(str(txt))
            if txt_noised is not None and txt_noised != txt:
                return add_joiners(txt_noised, starts_with_joiner, ends_with_joiner), 'homophone'

        elif next_error == 'misspell':
            txt_noised, type_misspell = misspell(str(txt))
            if txt_noised is not None and txt_noised != txt:
                return add_joiners(txt_noised, starts_with_joiner, ends_with_joiner), 'misspell'+type_misspell

        elif next_error == 'case':
            txt_noised, type_case = case(str(txt))
            if txt_noised is not None and txt_noised != txt:
                return add_joiners(txt_noised, starts_with_joiner, ends_with_joiner), 'case'+type_case

        elif next_error == 'hyphen':
            txt_noised, type_hyphen = hyphen(str(txt_clean), prev_tok, post_tok)
            if txt_noised is not None:
                return txt_noised, 'hyphen'+type_hyphen

        elif next_error == 'space':
            txt_noised, type_space = space(str(txt_clean), prev_tok, post_tok)
            if txt_noised is not None:
                return txt_noised, 'space'+type_space

        elif next_error == 'duplicate':
            txt_noised = duplicate(str(txt_clean), prev_tok, post_tok)
            if txt_noised is not None:
                return txt_noised, 'duplicate'

            
    return txt_clean, None
    
def noise_sentence(toks, args, stats, misspell, grammar, homophone, case, hyphen, space, duplicate):
    min_noises = int(len(toks)*args.min_r)
    max_noises = int(len(toks)*args.max_r)
    n_noises = random.randint(min_noises, max_noises) #[min_noises,max_noises] both included
    idxs = [idx for idx in range(len(toks))]
    random.shuffle(idxs)
    n_errors = 0
    errors = ['-'] * len(toks)
    toks_noisy = toks[:]
    for idx in idxs:
        if n_errors == n_noises:
            break
        prev_tok = toks[idx-1] if idx > 0 else None
        post_tok = toks[idx+1] if idx < len(toks)-1 else None
        tok, error = noise_token(toks[idx], prev_tok, post_tok, args, stats, misspell, grammar, homophone, case, hyphen, space, duplicate)
        if error is not None:
            if error.startswith('hyphen') or error.startswith('space'):
                logging.info('[{}] {} {} {} ---> {} {} {}'.format(error, prev_tok, toks[idx], post_tok, prev_tok, tok, post_tok))
            else:
                logging.info('[{}] {} ---> {}'.format(error, toks[idx], tok))
            n_errors += 1
            toks_noisy[idx] = tok
            errors[idx] = error
            stats[error] += 1
    return toks_noisy, errors, n_errors

#########################################################################################################
#########################################################################################################
#########################################################################################################

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=str, default=None, help='output file (for noisy output)')
    parser.add_argument('--grammar', type=str, default=None, help='grammar error file')
    parser.add_argument('--homophone', type=str, default=None, help='homophone error file')
    parser.add_argument('--min_r', type=float, default=0., help='Minimum ratio of noises/words per sentence (0.)')
    parser.add_argument('--max_r', type=float, default=0.4, help='Maximum ratio of noises/words per sentence (0.25)')
    parser.add_argument('--seed', type=int, default=0,    help='Seed for randomness (0)')    
    group_weights = parser.add_argument_group("Noise weights")
    group_weights.add_argument('--w_grammar', type=int, default=100, help='Weight for GRAMMAR noise (100)')
    group_weights.add_argument('--w_homophone', type=int, default=5, help='Weight for HOMOPHONE noise (5)')
    group_weights.add_argument('--w_hyphen', type=int, default=100, help='Weight for HYPHEN noise (100)')
    group_weights.add_argument('--w_misspell', type=int, default=1, help='Weight for MISSPELL noise (1)')
    group_weights.add_argument('--w_case', type=int, default=1, help='Weight for CASE noise (1)')
    group_weights.add_argument('--w_space', type=int, default=1, help='Weight for SPACE noise (1)')
    group_weights.add_argument('--w_duplicate', type=int, default=1, help='Weight for DUPLICATE noise (1)')
    args = parser.parse_args()
    if args.max_r == 0:
        args.max_r = 1.
    if args.o is not None:
        logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO', None), filename=args.o+'.log')
    else:
        logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO', None))
    logging.info("Options = {}".format(args.__dict__))
    onmttok = pyonmttok.Tokenizer("aggressive", joiner_annotate=True, joiner=ONMTTOK_JOINER)
    misspell = Misspell(wdelete=1,wrepeat=1,wexchange=1,wswap=1,wdiacritics=10,wconsd=25,wphone=50)
    grammar = Replacement(args.grammar) if args.grammar is not None else None
    homophone = Replacement(args.homophone) if args.homophone is not None else None
    case = Case()
    hyphen = Hyphen()
    space = Space()
    duplicate = Duplicate()
    
    if args.o is not None:
        fdo_noisy = open(args.o, 'w')
    
    tic = time.time()
    n_tokens = 0
    n_tokens_noised = 0
    n_sentences = 0
    n_sentences_noised = 0
    stats = defaultdict(int)
    for txt_clean in sys.stdin:
        n_sentences += 1
        txt_clean = txt_clean.rstrip()
        tok_clean = onmttok(txt_clean)
        tok_noisy, errors, n_errors = noise_sentence(tok_clean, args, stats, misspell, grammar, homophone, case, hyphen, space, duplicate)
        for i in range(len(tok_clean)):
            print(tok_clean[i] + '\t' + tok_noisy[i] + '\t' + errors[i])
        txt_noisy = onmttok.detokenize(tok_noisy)
        if args.o is not None:
            fdo_noisy.write(txt_noisy + '\n')
        else:
            print("{}".format(txt_noisy))
        #logging.info("nsentence:{} {} errors over {} tokens, {:.2f}".format(n_sentences, n_errors, len(tok_clean), 100.0*n_errors/len(tok_clean)))
        n_tokens += len(tok_noisy)
        n_tokens_noised += n_errors
        n_sentences_noised += 1 if n_errors else 0
        if n_sentences % 100000 == 0:
            logging.info('[{} sentences] noises {}'.format(n_sentences,stats.items()))
    toc = time.time()
    if args.o is not None:
        #fdo_clean.close()
        fdo_noisy.close()
        
    logging.info('Done {:.3f} seconds {:.1f} sentences/sec {:.1f} tokens/sec'.format(toc-tic,n_sentences/(toc-tic), n_tokens/(toc-tic)))
    logging.info('noised {:.2f} % of {} sentences'.format(100.0*n_sentences_noised/n_sentences,n_sentences))
    logging.info('noised {:.2f} % of {} tokens'.format(100.0*n_tokens_noised/n_tokens,n_tokens))
    logging.info('noises {}'.format(["{}:{}[{:.2f}%]".format(k,v,100.0*v/n_tokens_noised) for k,v in sorted(stats.items(),key=lambda item: item[1], reverse=True) ]))

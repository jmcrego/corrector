import sys
import time
import logging
import argparse
from collections import defaultdict
from Noises import Noise

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=str, default=None, help='output prefix file (o.noisy AND o.noisy_tok are built)', required=True)
    parser.add_argument('--grammar', type=str, default=None, help='grammar error file')
    parser.add_argument('--homophone', type=str, default=None, help='homophone error file')
    parser.add_argument('--min_r', type=float, default=0., help='Minimum ratio of noises/words per sentence (0.)')
    parser.add_argument('--max_r', type=float, default=0.4, help='Maximum ratio of noises/words per sentence (0.4)')
    parser.add_argument('--seed', type=int, default=0,    help='Seed for randomness (0)')    
    group_weights = parser.add_argument_group("Noise weights")
    group_weights.add_argument('--w_grammar', type=int, default=100, help='Weight for GRAMMAR noise (100)')
    group_weights.add_argument('--w_homophone', type=int, default=5, help='Weight for HOMOPHONE noise (5)')
    group_weights.add_argument('--w_hyphen', type=int, default=1, help='Weight for HYPHEN noise (100)')
    group_weights.add_argument('--w_misspell', type=int, default=1, help='Weight for MISSPELL noise (1)')
    group_weights.add_argument('--w_case', type=int, default=1, help='Weight for CASE noise (1)')
    group_weights.add_argument('--w_space', type=int, default=1, help='Weight for SPACE noise (1)')
    group_weights.add_argument('--w_duplicate', type=int, default=1, help='Weight for DUPLICATE noise (1)')
    misspell_weights = parser.add_argument_group("Misspell weights")
    misspell_weights.add_argument('--w_misspell_delete', type=int, default=1, help='Weight for MISSPELL:DELETE noise (1)')
    misspell_weights.add_argument('--w_misspell_repeat', type=int, default=1, help='Weight for MISSPELL:REPEAT noise (1)')
    misspell_weights.add_argument('--w_misspell_close', type=int, default=1, help='Weight for MISSPELL:CLOSE noise (1)')
    misspell_weights.add_argument('--w_misspell_swap', type=int, default=1, help='Weight for MISSPELL:SWAP noise (1)')
    misspell_weights.add_argument('--w_misspell_diacritics', type=int, default=10, help='Weight for MISSPELL:DIACRITICS noise (10)')
    misspell_weights.add_argument('--w_misspell_consd', type=int, default=25, help='Weight for MISSPELL:CONSD noise (25)')
    misspell_weights.add_argument('--w_misspell_phone', type=int, default=50, help='Weight for MISSPELL:PHONE noise (50)')

    args = parser.parse_args()
    #if args.o is not None:
    #    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO', None), filename=args.o+'.log')
    #else:
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO', None))
    logging.info("Options = {}".format(args.__dict__))

    n = Noise(args)
    if args.o is not None:
        fdo_noisy = open(args.o+'.noisy', 'w')
        fdo_noisy_tok = open(args.o+'.noisy_tok', 'w')

    tic = time.time()
    n_tokens = 0
    n_tokens_noisy = 0
    n_sentences = 0
    n_sentences_noisy = 0
    error_stats = defaultdict(int)
    for l in sys.stdin:
        tok_clean, tok_noisy, errors, txt_noisy = n(l.rstrip())
        fdo_noisy.write(txt_noisy + '\n')
        
        sentence_with_errors = False
        for i in range(len(tok_noisy)):
            fdo_noisy_tok.write(tok_clean[i] + '\t' + tok_noisy[i] + '\t' + errors[i] + '\n')
            n_tokens += 1
            if errors[i] != '-':
                n_tokens_noisy += 1
                sentence_with_errors = True
                error_stats[errors[i]] += 1
        n_sentences += 1
        n_sentences_noisy += 1 if sentence_with_errors else 0

        if n_sentences % 100000 == 0:
            logging.info('[{} sentences] noises {}'.format(n_sentences,error_stats.items()))
    toc = time.time()
    fdo_noisy.close()
    fdo_noisy_tok.close()
        
    logging.info('Done {:.3f} seconds {:.1f} sentences/sec {:.1f} tokens/sec'.format(toc-tic,n_sentences/(toc-tic), n_tokens/(toc-tic)))
    logging.info('Noised {:.2f} % of {} sentences'.format(100.0*n_sentences_noisy/n_sentences,n_sentences))
    logging.info('Noised {:.2f} % of {} tokens'.format(100.0*n_tokens_noisy/n_tokens,n_tokens))
    logging.info('Noises {}'.format(["{}={}[{:.2f}%]".format(k,v,100.0*v/n_tokens_noisy) for k,v in sorted(error_stats.items(),key=lambda item: item[1], reverse=True) ]))

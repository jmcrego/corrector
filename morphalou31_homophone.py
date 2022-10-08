import re
import sys
#import pickle
import logging
import argparse
from collections import defaultdict

def rewrite_wrd(wrd):
    if len(wrd) > 3:
        wrd = re.sub('^se ', '', wrd)
    if len(wrd) > 2:
        wrd = re.sub('^s\'', '', wrd)
    if ' ' in wrd or len(wrd) == 0:
        wrd = None
    return wrd

def rewrite_pho(pho):
    if pho == '':
        return None
    phos = set()
    for p in pho.split(' OU '):
        p = p.replace(' ','').upper()
        p = p.replace('/','')
        if len(p):
            phos.add(p)
    return list(phos)

def equivalents_of(pho, txt2phos, pho2txts):
    equivalents = set()
    equivalents.add(pho)
    while True:
        n_equivalents = len(equivalents)
        for p in list(equivalents):
            for txt in pho2txts[p]:
                for pho in txt2phos[txt]:
                    equivalents.add(pho)
        if len(equivalents) == n_equivalents:
            return equivalents

def find_equivalents_classes(pho2txts, txt2phos):
    key2equivalents = defaultdict(set)
    seen = set() 
    for pho in pho2txts:
        if pho in seen:
            continue
        equivalents = equivalents_of(pho, txt2phos, pho2txts) #equivalents contain for key pho
        key2equivalents[pho] = equivalents
        for p in equivalents:
            seen.add(p)
    #key of each pho
    pho2key = defaultdict(str)
    for key, phos in key2equivalents.items():
        for pho in phos:
            pho2key[pho] = key
    logging.info('Found {} classes'.format(len(key2equivalents)))

    txt2key = defaultdict(str)
    key2txts = defaultdict(set)
    for txt, phos in txt2phos.items():
        phos = list(phos)
        key = pho2key[phos[0]]
        for pho in phos:
            key2 = pho2key[pho]
            if key != key2:
                logging.error("Multiple keys {} --- {} in entry {}".format(key,key2,phos))
        txt2key[txt] = key
        key2txts[key].add(txt)

    txt2txts = defaultdict(list)
    for txt,key in txt2key.items():
        txts = list(key2txts[key])
        if txt in txts: ### should always happen
            txts.remove(txt)
        if len(txts):
            txt2txts[txt] = txts
    logging.info('Found {} txt2txts'.format(len(txt2txts)))
    return txt2txts
    
################################################
### MAIN #######################################
################################################

if __name__ == '__main__':

    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO', None)) #INFO = 20
    parser = argparse.ArgumentParser(description='This script creates the list of homophones for French words in Morphalou3.1_CSV (https://www.ortolang.fr/market/lexicons/morphalou).')
    parser.add_argument("fmorphalou", type=str, help="path to Morphalou3.1_CSV.csv")
    args = parser.parse_args()

    txt2phos = defaultdict(set)
    pho2txts = defaultdict(set)
    logging.info('Reading {}'.format(args.fmophalou))
    with open(args.fmorphalou,'r') as fdi:
        for n,l in enumerate(fdi):
            l = l.rstrip()
            toks = l.split(';') #0,7,9,16
            if len(toks) < 17 or toks[0] == 'LEMME' or toks[0] == 'GRAPHIE':
                logging.debug('filtered dsc\t{}'.format(l))
                continue
            
            txt = rewrite_wrd(toks[9])
            phos = rewrite_pho(toks[16])
            if txt is not None and phos is not None:
                for pho in phos:
                    txt2phos[txt].add(pho)
                    pho2txts[pho].add(txt)
                
            if toks[0]!=toks[9] and toks[7]!=toks[16]:
                txt = rewrite_wrd(toks[0])
                phos = rewrite_pho(toks[7])
                if txt is not None and phos is not None:
                    for pho in phos:
                        txt2phos[txt].add(pho)
                        pho2txts[pho].add(txt)
                
    logging.info('Found {} words {} phonetizations'.format(len(txt2phos),len(pho2txts)))
    txt2txts = find_equivalents_classes(pho2txts, txt2phos)
    with open(args.fmorphalou+'.homophones', 'w') as fdo:
        for txt,txts in txt2txts.items():
            fdo.write("{}\t{}\n".format(txt,'\t'.join(txts)))
    #with open(args.fmorphalou+'.homophones.pickle', 'wb') as fdo:
    #    pickle.dump(txt2txts, fdo, protocol=-1)
        


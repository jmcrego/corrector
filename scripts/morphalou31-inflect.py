import re
import sys
import random
#import pickle
import logging
import argparse
from collections import defaultdict

sep = '|'
WORD = re.compile(r'^\w+$')

class LemRules():
    
    def __init__(self, frules, txt2pos, txtpos2lem, lempos2txt, txtlempos2feats):
        self.txt2pos = txt2pos
        self.txtpos2lem = txtpos2lem
        self.lempos2txt = lempos2txt
        self.txtlempos2feats = txtlempos2feats
        self.rules = defaultdict(list)
        with open(frules,"r") as f:
            for l in f: #lemendswith=er,pos=VER,tense=infinitive;lemendswith=er,pos=VER,mode=participle,tense=past
                l = l.rstrip()
                if l.startswith('#') or len(l) == 0:
                    continue
                toks = l.split(';')
                #print(toks)
                for i in range(len(toks)):
                    left = toks[i]  #"lemendswith=er,pos=VER,tense=infinitive"
                    right = toks[:] #["lemendswith=er,pos=VER,tense=infinitive", "lemendswith=er,pos=VER,mode=participle,tense=past"]
                    #right.pop(i)
                    self.rules[left] = right
        logging.info('Read {} with {} rules'.format(frules,len(self.rules)))

    def match(self, siderule, pos, lem, mode, tense, pers, nombre, genre):
        #print("siderule={}".format(siderule))
        #print("match pos={} lem={} mode={} tense={} pers={} nombre={} genre={} siderule={}".format(pos, lem, mode, tense, pers, nombre, genre, siderule))
        for condition in siderule.split(','): #lemendswith=er,pos=VER,tense=infinitive
            #print("condition={}")
            csrc,ctgt = condition.split('=')
            if csrc == 'lemendswith':
                if not lem.endswith(ctgt):
                    return False
            elif csrc == 'pos':
                if not ctgt == pos:
                    return False
            elif csrc == 'tense':
                if not ctgt == tense:
                    return False
            elif csrc == 'mode':
                if not ctgt == mode:
                    return False
            elif csrc == 'pers':
                if not ctgt == pers:
                    return False
            elif csrc == 'nombre':
                if not ctgt == nombre:
                    return False
            elif csrc == 'genre':
                if not ctgt == genre:
                    return False
        return True

    def __call__(self, itxt, ilem, ipos, feats):
        ipos,ilem,imode,itense,ipers,inombre,igenre = feats.split(sep)
        #print('Looking for errors of:')
        #print('itxt: {}'.format(itxt))
        #print('ipos: {}'.format(ipos))
        #print('ilem: {}'.format(ilem))
        #print('imode: {}'.format(imode))
        #print('itense: {}'.format(itense))
        #print('ipers: {}'.format(ipers))
        #print('inombre: {}'.format(inombre))
        #print('igenre: {}'.format(igenre))
        ltxt = [] #found words and their features
        for leftrule, rightrules in self.rules.items():
            if self.match(leftrule, ipos, ilem, imode, itense, ipers, inombre, igenre):
                #print('\tmatch\t{}'.format(leftrule))
                for rightrule in rightrules:
                    #print('\t\t{}'.format(rightrule))
                    for rtxt in self.lempos2txt[ilem+sep+ipos]: ### replacement rtxt must have the same lem AND pos than itxt
                        #print('\t\t\trtxt: {}'.format(rtxt))
                        if rtxt != itxt: ### must be different
                            for rfeats in self.txtlempos2feats[rtxt+sep+ilem+sep+ipos]:
                                #print('\t\t\t\trfeats: {}'.format(rfeats))
                                rpos, rlem, rmode, rtense, rpers, rnombre, rgenre = rfeats.split(sep)
                                if rlem == ilem and rpos == ipos: ### must be same lem AND pos
                                    if self.match(rightrule, rpos, rlem, rmode, rtense, rpers, rnombre, rgenre):
                                        #logging.debug('\trmatch\t{}\t{}'.format(rplm,rightrule))
                                        ltxt.append(rtxt)
        return ltxt


def rewrite_pos(pos,spos,l):
    #     40 Conjonction;coordination
    #    139 Conjonction;subordination
    #      4 Déterminant;défini
    #      5 Déterminant;démonstratif
    #      1 Déterminant;exclamatif
    #     40 Déterminant;indéfini
    #      7 Déterminant;possessif
    #      9 Pronom;démonstratif
    #     38 Pronom;indéfini
    #     12 Pronom;interrogatif
    #     42 Pronom;personnel
    #     11 Pronom;possessif
    #      9 Pronom;relatif
    #      2 Verbe;auxiliaire
    #     30 Verbe;défectif
    #     18 Verbe;impersonnel
    #  36523 Adjectif qualificatif
    #   4157 Adverbe
    #    179 Conjonction
    #     57 Déterminant
    #    422 Interjection
    #    198 Nombre
    # 102238 Nom commun
    #    258 Préposition
    #    121 Pronom
    #  14762 Verbe

    if pos == 'Verbe':
        pos = 'VER'
    elif pos == 'Préposition':
        pos = 'PRE'
    elif pos == 'Adverbe':
        pos = 'ADV'
    elif pos == 'Conjonction':
        pos = 'CON'
    elif pos == 'Déterminant':
        pos = 'ART'
        if spos == 'défini':
            pos = 'ART:DEF'
        if spos == 'démonstratif':
            pos = 'ART:DEM'
        if spos == 'exclamatif':
            pos = 'ART:EXC'
        if spos == 'indéfini':
            pos = 'ART:IND'
        if spos == 'possessif':
            pos = 'ART:POS'
    elif pos == 'Interjection':
        pos = 'INT'
    elif pos == 'Pronom':
        pos = 'PRO'
        if spos == 'démonstratif':
            pos = 'PRO:DEM'
        elif spos == 'défini':
            pos = 'PRO:IND'
        elif spos == 'interrogatif':
            pos = 'PRO:INT'
        elif spos == 'personnel':
            pos = 'PRO:PER'
        elif spos == 'possessif':
            pos = 'PRO:POS'
        elif spos == 'relatif':
            pos = 'PRO:REL'                
    elif pos == 'Nombre':
        pos = 'ADJ'
    elif pos == 'Adjectif qualificatif':
        pos = 'ADJ'
    elif pos == 'Nom commun':
        pos = 'NOM'
    else:
        logging.debug('UNPARSED pos [{}] {}'.format(pos,l))
        pos = None
    return pos

def rewrite_lem(lem):
    if len(lem) > 3:
        lem = re.sub('^se ', '', lem)
    if len(lem) > 2:
        lem = re.sub('^s\'', '', lem)
    #lem = lem.replace('se ','').replace('s\'','')
    if len(lem) == 0 or ' ' in lem:
        #logging.debug('UNPARSED lem [{}] {}'.format(lem,l))
        lem = None
    return lem

def rewrite_wrd(wrd):
    if len(wrd) > 3:
        wrd = re.sub('^se ', '', wrd)
    if len(wrd) > 2:
        wrd = re.sub('^s\'', '', wrd)
    if ' ' in wrd or len(wrd) == 0:
        wrd = None
    return wrd

def rewrite_pers(pers):
    if pers == 'firstPerson':
        return '1'
    if pers == 'secondPerson':
        return '2'
    if pers == 'thirdPerson':
        return '3'
    if pers == '-':
        return '-'
    sys.stderr.write('pers: {}\n'.format(pers))
    return pers

def rewrite_nombre(nombre):
    if nombre == 'singular':
        return 'Sing'
    if nombre == 'plural':
        return 'Plur'
    if nombre == 'invariable':
        return 'Inv'
    if nombre == '-':
        return '-'
    sys.stderr.write('nombre: {}\n'.format(nombre))
    return nombre

def rewrite_genre(genre):
    if genre == 'feminine':
        return 'Fem'
    if genre == 'masculine':
        return 'Masc'
    if genre == 'invariable':
        return 'Inv'
    if genre == 'neuter':
        return '-'
    if genre == '-':
        return '-'
    sys.stderr.write('genre: {}\n'.format(genre))
    return genre

def add_flection(l,toks,txt2pos,txtpos2lem,lempos2txt,txtlempos2feats,base_lem,base_pos,base_spos,base_genre):
    #GRAPHIE;ID;NOMBRE;MODE;GENRE;TEMPS;PERSONNE;PHONÉTIQUE
    txt = rewrite_wrd(toks[0])

    if txt is None:
        logging.debug('filtered txt\t{}'.format(l))
        return False

    if WORD.match(txt) is None: #i dont want punctuation in words (ex: Mr. l' vis-à-vis ...)
        return False
    
    lem = rewrite_lem(base_lem)
    if lem is None:
        logging.debug('filtered lem\t{}'.format(l))
        return False
    pos = rewrite_pos(base_pos, base_spos, l)
    if pos is None:
        logging.debug('filtered pos\t{}'.format(l))
        return False

    nombre = rewrite_nombre(toks[2])
    mode   = toks[3]
    genre = rewrite_genre(toks[4]) if toks[4] != '-' else rewrite_genre(base_genre)
    temps  = toks[5]
    pers   = rewrite_pers(toks[6])

    if lem in ['Pa', 'Bq', 'pa', 'lm', 'Hz', 'cd', 'Wh', 'gray', 'VA', 'sr', 'Da', 'S', 'J', 'A', 'C', 'm', 's', 'l', 'L', 'T', 'N', 'H', 'F', 'G', 'g', 'K', 'V', 'W', 'Ci', 'eV', 'Gy', 'lx', 'Sv', 'kat', 'Wb', 'Ω', 'mètre', 'pascal', 'watt', 'gramme', 'calorie', 'hertz', 'octet', 'ampère', 'gauss', 'newton', 'litre', 'cal', 'coulomb', 'curie', 'henry', 'joule', 'kelvin', 'lumen', 'mole', 'ohm', 'radian', 'roentgen', 'tgen', 'röntgens', 'seconde', 'siemens', 'stéradian', 'tesla', 'var', 'volt', 'voltampère', 'wéber', 'weber', 'lux', 'mol', 'röntgen', 'rad', 'katal', 'wattheure', 'farad', 'candéla', 'candela', 'sievert', 'becquerel', 'électronvolts', 'électronvolt', 'dalton', 'daltons']:  #i dont want units and their derivates
        return False
    
    txt2pos[txt].add(pos)
    txtpos2lem[txt+sep+pos].add(lem)
    lempos2txt[lem+sep+pos].add(txt)
    txtlempos2feats[txt+sep+lem+sep+pos].add( pos+sep+lem+sep+mode+sep+temps+sep+pers+sep+nombre+sep+genre )
    #fdo.write('\t'.join([txt, lem, sep.join([pos, mode, temps, pers, nombre, genre])])+'\n')
    return True

def parse_morphalou(fin):
    txt2pos = defaultdict(set)
    txtpos2lem = defaultdict(set)
    lempos2txt = defaultdict(set)
    txtlempos2feats = defaultdict(set)
    
    lem_str = ''
    pos_str = ''
    spos_str = ''
    genre_str = ''
    n = 0
    m = 0
    logging.info('Reading {}'.format(fin))
    with open(fin,'r') as fdi:
        for l in fdi:
            n += 1
            l = l.rstrip()
            toks = l.split(';')

            if len(toks) < 17 or toks[0] == 'LEMME' or toks[0] == 'GRAPHIE':
                logging.debug('filtered\t{}'.format(l))
                continue

            if toks[0] != '': ### this is a base form, keep some fields to be used with its flections
                #GRAPHIE;ID;CATÉGORIE;SOUS CATÉGORIE;LOCUTION;GENRE;AUTRES LEMMES LIÉS;PHONÉTIQUE
                lem_str = toks[0]
                pos_str = toks[2]
                spos_str = toks[3]
                genre_str = toks[5]

            ### add entry
            if add_flection(l,toks[9:17],txt2pos,txtpos2lem,lempos2txt,txtlempos2feats,lem_str,pos_str,spos_str,genre_str):
                m += 1
                    
    logging.info('Loaded Lexicon with {} entries, output {} entries'.format(n,m))
    return txt2pos, txtpos2lem, lempos2txt, txtlempos2feats

################################################
### MAIN #######################################
################################################

if __name__ == '__main__':

    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO', None)) #INFO = 20
    parser = argparse.ArgumentParser(description='This script creates the list of grammatical errors detailed in rules for French words in Morphalou3.1_CSV (https://www.ortolang.fr/market/lexicons/morphalou).')
    parser.add_argument("fmorphalou", type=str, help="path to Morphalou3.1_CSV.csv")
    parser.add_argument('frules', type=str, default=None, help='grammar error rules config file')
    args = parser.parse_args()
    logging.info("Options = {}".format(args.__dict__))
    
    txt2pos, txtpos2lem, lempos2txt, txtlempos2feats = parse_morphalou(args.fmorphalou)
    rules = LemRules(args.frules, txt2pos, txtpos2lem, lempos2txt, txtlempos2feats)

    with open(args.fmorphalou + ".grammar", 'w') as fdo:
        for txt in txt2pos:
            stxt = set()
            for pos in txt2pos[txt]:
                for lem in txtpos2lem[txt+sep+pos]:
                    for feats in txtlempos2feats[txt+sep+lem+sep+pos]:
                        ltxt = rules(txt,lem,pos,feats)
                        for t in ltxt:
                            stxt.add(t)
            if len(stxt) > 1:
                fdo.write("{}\t{}\n".format(txt,'\t'.join(list(stxt))))
            if len(stxt) > 9:
                print("{}\t{}".format(txt,'\t'.join(list(stxt))))

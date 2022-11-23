import sys
import argparse
from collections import defaultdict

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This script outputs the set of words with given pos for French words in Morphalou3.1_CSV (https://www.ortolang.fr/market/lexicons/morphalou).')
    parser.add_argument("fmorphalou", type=str, help="path to Morphalou3.1_CSV.csv")
    parser.add_argument("--pos", nargs='+', default=[], help="list of pos to consider")
    args = parser.parse_args()

    seen = defaultdict(int)
    with open(args.fmorphalou,'r') as fdi:
        for n,l in enumerate(fdi):
            l = l.rstrip()
            toks = l.split(';')

            if len(toks) < 17 or toks[0] == 'LEMME' or toks[0] == 'GRAPHIE':
                continue

            if toks[0] != '':
                txt = toks[0]
                pos = toks[2]
                if pos == '':
                    continue
                n = pos.find(' ')
                if n>=0:
                    pos = pos[:n]
            else:
                txt = toks[9]

            if not txt.isalpha():
                continue
            
            if pos in args.pos and txt not in seen:
                print("{}".format(txt))
                seen[txt] += 1
                

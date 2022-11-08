import random
from collections import defaultdict

def match_case(txt, as_txt):
    if as_txt.isupper():
        return txt.upper()
    elif as_txt.islower():
        return txt
    elif as_txt[0].isupper() and as_txt[1:].islower():
        return txt[0].upper() + txt[1:]
    return txt

class Replacement():
    def __init__(self,f,rep_type,min_error_length=1):
        self.rep_type = rep_type
        self.replacements = defaultdict(list)
        self.min_error_length = min_error_length
        with open(f,'r') as fd:
            for l in fd:
                toks = l.rstrip().split('\t')
                self.replacements[toks[0]] = toks[1:]

    def __call__(self,txt):
        if len(txt) < self.min_error_length:
            return '', ''
        txt_lc = txt.lower()
        if txt_lc in self.replacements:
            txts_lc = self.replacements[txt_lc]
            if len(txts_lc) == 0:
                return '', ''
            if len(txts_lc) > 1:
                random.shuffle(txts_lc)
            txt_new = match_case(txts_lc[0], txt)
            return txt_new, self.rep_type
        return '', ''

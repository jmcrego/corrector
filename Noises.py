import re
#import sys
import copy
import random
import logging
import pyonmttok
import unicodedata
from collections import defaultdict

ONMTTOK_JOINER = '￭'
WORD = r'[\w]+'
PATTERN_WORD = re.compile(WORD)
CONSONNES_DOUBLES = 'dcflsptmnrgDCFLSPTMNRG' #bkz 
ACCENTS = 'aaàáâäeeéèêëiiíìîïooóòôöuuúùûüAAÀÁÂÄEEÉÈÊËIIÍÌÎÏOOÓÒÔÖUUÚÙÛÜ'
ACCENTS_FREQ = { 'a': {'a':20, 'á':1, 'à':10, 'â':1, 'ä':0}, 'e': {'e':20, 'é':10, 'è':9, 'ê':2, 'ë':1}, 'i': {'i':20, 'í':3, 'ì':0, 'î':1, 'ï':1}, 'o': {'o':20, 'ó':0, 'ò':0, 'ô':2, 'ö':1}, 'u': {'u':20, 'ú':0, 'ù':2, 'û':1, 'ü':1}}
#KEYBOARD = {'1': '`2qw', '2': '13qweèéêë', '3': '24weèéêëqr', '4': '35eèéêërwt', '5': '46rteèéêëy', '6': '57tyruùúûü', '7': '68yuùúûütiìíîï', '8': '79uùúûüiìíîïyoòóôö', '9': '80iìíîïoòóôöuùúûüp', '0': '9-oòóôöpiìíîï[', '-': '0=p[oòóôö]', '=': '-[]p\\', 'q': 'waàáâä12s`3', 'w': 'qeèéêës23aàáâäd14', 'e': 'èéêëwrd34sf25aàáâä', 'r': 'eèéêëtf45dg36s', 't': 'ryg56fh47d', 'y': 'tuùúûüh67gj58f', 'u': 'ùúûüyiìíîïj78hk69g', 'i': 'ìíîïuùúûüoòóôök89jl7', 'o': 'òóôöiìíîïpl90k;8-j', 'p': "oòóôö[;0-l'9=k", '[': "p]'-=;0l", ']': "[\\='-;", '\\': "]='", 'a': 'àáâäszqwxeèéêë', 's': 'aàáâädxweèéêëzcçqrf', 'd': 'fscçeèéêërxvwtzg', 'f': 'dgvrtcçbeèéêëyxs', 'g': 'fhbtyvnruùúûücçd', 'h': 'gjnyuùúûübmtiìíîïv', 'j': 'hkmuùúûüiìíîïn,yoòóô', 'k': 'jl,iìíîïoòóôöm.uùúûü', 'l': 'k;.oòóôöp,/iìíîï[m', ';': "l'/p[.oòóôö],", "'": ';[]/p\\.', 'z': 'xaàáâäsd', 'x': 'zcçsdaàáâäf', 'c': 'çvxdfsg', 'v': 'cçbfgdh', 'b': 'vnghfj', 'n': 'bmhjgk', 'm': 'n,jkhl', ',': 'm.klj;', '.': ",/l;k'", '/': ".;'l", '~': '!Q', '!': '~@QW', '@': '!#QWEÈÉÊË', '#': '@$WEÈÉÊËQR', '$': '#%EÈÉÊËRWT', '%': '$^RTEÈÉÊËY', '^': '%&TYRUÙÚÛÜ', '&': '^*YUÙÚÛÜTIÌÍÎÏ', '*': '&(UÙÚÛÜIÌÍÎÏYOÒÓÔÖ', '(': '*)IÌÍÎÏOÒÓÔÖUÙÚÛÜP', ')': '(_OÒÓÔÖPIÌÍÎÏ{', '_': ')+P{OÒÓÔÖ}', '+': '_{}P|', 'Q': 'WAÀÁÂÄ!@S~#', 'W': 'QEÈÉÊËS@#AÀÁÂÄD!$', 'E': 'ÈÉÊËWRD#$SF@%AÀÁÂÄ', 'R': 'EÈÉÊËTF$%DG#^S', 'T': 'RYG%^FH$&D', 'Y': 'TUÙÚÛÜH^&GJ%*F', 'U': 'ÙÚÛÜYIÌÍÎÏJ&*HK^(G', 'I': 'ÌÍÎÏUÙÚÛÜOÒÓÔÖK*(JL&', 'O': 'ÒÓÔÖIÌÍÎÏPL()K:*_J', 'P': 'OÒÓÔÖ{:)_L"(+K', '{': 'P}"_+:)L', '}': '{|+"_:', '|': '}+"', 'A': 'ÀÁÂÄSZQWXEÈÉÊË', 'S': 'AÀÁÂÄDXWEÈÉÊËZCÇQRF', 'D': 'FSCÇEÈÉÊËRXVWTZG', 'F': 'DGVRTCÇBEÈÉÊËYXS', 'G': 'FHBTYVNRUÙÚÛÜCÇD', 'H': 'GJNYUÙÚÛÜBMTIÌÍÎÏV', 'J': 'HKMUÙÚÛÜIÌÍÎÏN<YOÒÓÔ', 'K': 'JL<IÌÍÎÏOÒÓÔÖM>UÙÚÛÜ', 'L': 'K:>OÒÓÔÖP<?IÌÍÎÏ{M', ':': 'L"?P{>OÒÓÔÖ}<', '"': ':{}?P|>', 'Z': 'XAÀÁÂÄSD', 'X': 'ZCÇSDAÀÁÂÄF', 'C': 'ÇVXDFSG', 'V': 'CÇBFGDH', 'B': 'VNGHFJ', 'N': 'BMHJGK', 'M': 'N<JKHL', '<': 'M>KLJ:', '>': '<?L:K"', '?': '>:"L'}
KEYBOARD_PUNCT = {'-': '0=p[oòóôö]', '=': '-[]p\\', '[': "p]'-=;0l", ']': "[\\='-;", '\\': "]='", ';': "l'/p[.oòóôö],", "'": ';[]/p\\.', ',': 'm.klj;', '.': ",/l;k'", '/': ".;'l", '~': '!Q', '!': '~@QW', '@': '!#QWEÈÉÊË', '#': '@$WEÈÉÊËQR', '$': '#%EÈÉÊËRWT', '%': '$^RTEÈÉÊËY', '^': '%&TYRUÙÚÛÜ', '&': '^*YUÙÚÛÜTIÌÍÎÏ', '*': '&(UÙÚÛÜIÌÍÎÏYOÒÓÔÖ', '(': '*)IÌÍÎÏOÒÓÔÖUÙÚÛÜP', ')': '(_OÒÓÔÖPIÌÍÎÏ{', '_': ')+P{OÒÓÔÖ}', '+': '_{}P|', 'Q': 'WAÀÁÂÄ!@S~#', '{': 'P}"_+:)L', '}': '{|+"_:', '|': '}+"', ':': 'L"?P{>OÒÓÔÖ}<', '"': ':{}?P|>', '<': 'M>KLJ:', '>': '<?L:K"', '?': '>:"L'}
KEYBOARD_LETTER = {'1': '`2qw', '2': '13qweèéêë', '3': '24weèéêëqr', '4': '35eèéêërwt', '5': '46rteèéêëy', '6': '57tyruùúûü', '7': '68yuùúûütiìíîï', '8': '79uùúûüiìíîïyoòóôö', '9': '80iìíîïoòóôöuùúûüp', '0': '9-oòóôöpiìíîï[', 'q': 'waàáâä12s`3', 'w': 'qeèéêës23aàáâäd14', 'e': 'èéêëwrd34sf25aàáâä', 'r': 'eèéêëtf45dg36s', 't': 'ryg56fh47d', 'y': 'tuùúûüh67gj58f', 'u': 'ùúûüyiìíîïj78hk69g', 'i': 'ìíîïuùúûüoòóôök89jl7', 'o': 'òóôöiìíîïpl90k;8-j', 'p': "oòóôö[;0-l'9=k", 'a': 'àáâäszqwxeèéêë', 's': 'aàáâädxweèéêëzcçqrf', 'd': 'fscçeèéêërxvwtzg', 'f': 'dgvrtcçbeèéêëyxs', 'g': 'fhbtyvnruùúûücçd', 'h': 'gjnyuùúûübmtiìíîïv', 'j': 'hkmuùúûüiìíîïn,yoòóô', 'k': 'jl,iìíîïoòóôöm.uùúûü', 'l': 'k;.oòóôöp,/iìíîï[m', 'z': 'xaàáâäsd', 'x': 'zcçsdaàáâäf', 'c': 'çvxdfsg', 'v': 'cçbfgdh', 'b': 'vnghfj', 'n': 'bmhjgk', 'm': 'n,jkhl', 'Q': 'WAÀÁÂÄ!@S~#', 'W': 'QEÈÉÊËS@#AÀÁÂÄD!$', 'E': 'ÈÉÊËWRD#$SF@%AÀÁÂÄ', 'R': 'EÈÉÊËTF$%DG#^S', 'T': 'RYG%^FH$&D', 'Y': 'TUÙÚÛÜH^&GJ%*F', 'U': 'ÙÚÛÜYIÌÍÎÏJ&*HK^(G', 'I': 'ÌÍÎÏUÙÚÛÜOÒÓÔÖK*(JL&', 'O': 'ÒÓÔÖIÌÍÎÏPL()K:*_J', 'P': 'OÒÓÔÖ{:)_L"(+K', 'A': 'ÀÁÂÄSZQWXEÈÉÊË', 'S': 'AÀÁÂÄDXWEÈÉÊËZCÇQRF', 'D': 'FSCÇEÈÉÊËRXVWTZG', 'F': 'DGVRTCÇBEÈÉÊËYXS', 'G': 'FHBTYVNRUÙÚÛÜCÇD', 'H': 'GJNYUÙÚÛÜBMTIÌÍÎÏV', 'J': 'HKMUÙÚÛÜIÌÍÎÏN<YOÒÓÔ', 'K': 'JL<IÌÍÎÏOÒÓÔÖM>UÙÚÛÜ', 'L': 'K:>OÒÓÔÖP<?IÌÍÎÏ{M', 'Z': 'XAÀÁÂÄSD', 'X': 'ZCÇSDAÀÁÂÄF', 'C': 'ÇVXDFSG', 'V': 'CÇBFGDH', 'B': 'VNGHFJ', 'N': 'BMHJGK', 'M': 'N<JKHL'}

def change_accent(letter):
    if letter not in ACCENTS:
        return None
    letter_norm = unicodedata.normalize('NFD', letter)
    letter_norm = letter_norm.encode('ascii', 'ignore')
    letter_norm = letter_norm.decode("utf-8").lower() #letter without accent and lowercased
    if letter_norm not in ACCENTS_FREQ:
        return None
    options = ACCENTS_FREQ[letter_norm].copy() #ex: {'a':20, 'á':1, 'à':10, 'â':1, 'ä':0}
    if letter.lower() in options:
        options[letter.lower()] = 0
    new_letter = random.choices(list(options.keys()), list(options.values()), k=1)[0]
    if letter.isupper():
        new_letter = new_letter.upper()
    return new_letter

def change_keyboard(letter):
    if letter not in KEYBOARD_LETTER and letter not in KEYBOARD_PUNCT:
        return None
    near_letters = list(KEYBOARD_LETTER[letter]) if letter in KEYBOARD_LETTER else list(KEYBOARD_PUNCT[letter])
    if letter in near_letters:
        near_letters.remove(letter)
    random.shuffle(near_letters)
    return near_letters[0]    

class Misspell():
    #misspell_weights.add_argument('--w_misspell_delete', type=int, default=1, help='Weight for MISSPELL:DELETE noise (1)')
    #misspell_weights.add_argument('--w_misspell_repeat', type=int, default=1, help='Weight for MISSPELL:REPEAT noise (1)')
    #misspell_weights.add_argument('--w_misspell_close', type=int, default=1, help='Weight for MISSPELL:CLOSE noise (1)')
    #misspell_weights.add_argument('--w_misspell_swap', type=int, default=1, help='Weight for MISSPELL:SWAP noise (1)')
    #misspell_weights.add_argument('--w_misspell_diacritics', type=int, default=10, help='Weight for MISSPELL:DIACRITICS noise (10)')
    #misspell_weights.add_argument('--w_misspell_consd', type=int, default=25, help='Weight for MISSPELL:CONSD noise (25)')
    #misspell_weights.add_argument('--w_misspell_phone', type=int, default=50, help='Weight for MISSPELL:PHONE noise (50)')

    def __init__(self,args,min_word_length=2):
        self.wdelete = args.w_misspell_delete
        self.wrepeat = args.w_misspell_repeat
        self.wclose = args.w_misspell_close
        self.wswap = args.w_misspell_swap
        self.wdiacritics = args.w_misspell_diacritics
        self.wconsd = args.w_misspell_consd
        self.wphone = args.w_misspell_phone
        self.min_word_length = min_word_length
        logging.info('Built Misspell noiser')
        
    def __call__(self, word):
        words = []
        types = []
        weights = []
        if len(word) < self.min_word_length:
            return None, None
        
        for i in range(len(word)):
            
            ### delete char i
            if len(word) > 1:
                weights.append(self.wdelete)
                types.append('misspell:delete')
                words.append(word)
                words[-1] = words[-1][:i] + words[-1][i+1:]

            ### repeat char i
            weights.append(self.wrepeat)
            types.append('misspell:repeat')
            words.append(word)
            words[-1] = words[-1][:i+1] + words[-1][i:]

            ### swap chars i <=> i+1
            if i < len(word)-1:
                c1 = word[i]
                c2 = word[i+1]
                weights.append(self.wswap)
                types.append('misspell:swap')
                words.append(word)
                words[-1] = words[-1][:i] + c2 + c1 + words[-1][i+2:]

            ### replace char i by close to it
            near_letter = change_keyboard(word[i])
            if near_letter is not None:
                weights.append(self.wclose)
                types.append('misspell:close')
                words.append(word)
                words[-1] = words[-1][:i] + near_letter + words[-1][i+1:]

            ### replace char i if vowel by the same with diacritics
            new_letter = change_accent(word[i])
            if new_letter is not None:
                weights.append(self.wdiacritics)
                types.append('misspell:diacr')
                words.append(word)
                words[-1] = words[-1][:i] + new_letter + words[-1][i+1:]

            ### delete/add char i if doubled consonant
            if i>0 and i<len(word)-2 and word[i] == word[i+1] and word[i] in CONSONNES_DOUBLES:
                weights.append(self.wconsd)
                types.append('misspell:consd:del')
                words.append(word)
                words[-1] = words[-1][:i] + words[-1][i+1:] #char i is not added

            ### add double consonnant
            if i>0 and i<len(word)-1 and word[i] != word[i+1] and word[i] != word[i-1] and word[i] in CONSONNES_DOUBLES:
                weights.append(self.wconsd)
                types.append('misspell:consd:add')
                words.append(word)
                words[-1] = words[-1][:i+1] + words[-1][i:] #char i is added twice

            ### 'ph' -> 'f'
            if len(word)>i+1 and word[i] == 'p' and word[i+1] == 'h':
                weights.append(self.wphone)
                types.append('misspell:phone:ph2f')
                words.append(word)
                words[-1] = words[-1][:i] + 'f' + words[-1][i+2:]
                
            ### 'h' -> 
            if word[i] == 'h' and not (i>0 and word[i-1] in 'CcPp'):
                weights.append(self.wphone)
                types.append('misspell:phone:h2-')
                words.append(word)
                words[-1] = words[-1][:i] + words[-1][i+1:]

        if len(words) == 0:
            return None, None

        #return random.choices(words,weights)[0]
        i = random.choices([i for i in range(len(words))],weights)[0]
        return words[i], types[i]

    def is_spell(self, wrd1, wrd2):
        if abs(len(wrd1)-len(wrd2)) > 1:
            return False
        swrd1 = set(list(wrd1))
        swrd2 = set(list(wrd2))
        if len(swrd1-swrd2) <= 1 and len(swrd2-swrd1) <= 1:
            return True
        return False

    
class Case():
    def __init__(self):
        logging.info('Built Case noiser')

    def __call__(self, txt):
        if txt.isupper(): # IBM
            if len(txt) > 1 and random.random() < 0.5:
                return txt[0] + txt[1:].lower(), 'case:X' #Ibm
            else:
                return txt.lower(), 'case:X' # ibm
        
        if txt.islower(): #maison
            if len(txt) > 1 and random.random() < 0.5:
                return txt[0].upper() + txt[1:], 'case:x' #Maison
            else:
                return txt.upper(), 'case:x' #MAISON
        
        if txt[0].isupper() and txt[1:].islower(): #Table
            if random.random() < 0.5:
                return txt.upper(), 'case:Xx' #TABLE
            else:
                return txt.lower(), 'case:Xx' #table
        return None, None

class Hyphen():
    def __init__(self):
        logging.info('Built Hyphen noiser')

    def __call__(self, txt_tok, prev_tok, next_tok, prev_is_noisy, next_is_noisy):
        if not prev_is_noisy and not next_is_noisy and txt_tok == ONMTTOK_JOINER + '-' + ONMTTOK_JOINER and PATTERN_WORD.match(prev_tok) and PATTERN_WORD.match(next_tok):  #word_prev #-# word_next
            if random.random() < 0.5:
                return ONMTTOK_JOINER, 'hyphen:add:split' ### anti #-# globalization => # (curr is #-#)
            else:
                return '', 'hyphen:add:merge'             ### anti #-# globalization => '' (curr is #-#)
        elif PATTERN_WORD.match(txt_tok):
            if not prev_is_noisy and prev_tok is not None and PATTERN_WORD.match(prev_tok): ### anti globalization => #-# globalization (curr is globalization)
                return ONMTTOK_JOINER + '-' + ONMTTOK_JOINER + ' ' + txt_tok, 'hyphen:del:prev'
            if not next_is_noisy and next_tok is not None and PATTERN_WORD.match(next_tok): ### anti globalization => anti #-# (curr is anti)
                return txt_tok + ' ' + ONMTTOK_JOINER + '-' + ONMTTOK_JOINER, 'hyphen:del:next'
        return None, None
            
class Space():
    def __init__(self):
        logging.info('Built Space noiser')

    def __call__(self, txt_clean, prev_tok, next_tok): #txt_clean may contain joiners

        if not PATTERN_WORD.match(txt_clean):
            return None, None
        
        if random.random() < 0.5: ### split
            minlen = 3
            if len(txt_clean) < 2*minlen: #minimum length of resulting spltted tokens
                return None, None
            k = random.randint(minlen,len(txt_clean)-minlen)
            prev = txt_clean[:k]
            post = txt_clean[k:]
            return prev + " " + post, 'space:del'

        else: ### merge            
            if next_tok is not None and not next_tok.startswith(ONMTTOK_JOINER):
                return txt_clean + ONMTTOK_JOINER, 'space:add:next' ### merge with next
            if prev_tok is not None and not prev_tok.endswith(ONMTTOK_JOINER):
                return ONMTTOK_JOINER + txt_clean, 'space:add:prev' ### merge with prev
        return None, None

class Duplicate():
    def __init__(self):
        logging.info('Built Duplicate noiser')

    def __call__(self, txt_clean): #txt_clean may contain joiners
        if not PATTERN_WORD.match(txt_clean): #only duplicate words
            return None, None
        return txt_clean + ' ' + txt_clean, 'duplicate'
    
class Replacement():
    def __init__(self,f,rep_type,min_error_length=2):
        self.rep_type = rep_type
        self.replacements = defaultdict(list)
        self.min_error_length = min_error_length ### min error length
        with open(f,'r') as fd:
            for l in fd:
                toks = l.rstrip().split('\t')
                self.replacements[toks[0]] = toks[1:]
        logging.info('Loaded Replacements file with {} entries'.format(f,len(self.replacements)))
                
    def __call__(self,txt):
        if len(txt) < self.min_error_length: 
            return None, None
        txt_lc = txt.lower()
        if txt_lc in self.replacements:
            txts_lc = self.replacements[txt_lc]
            if len(txts_lc) > 1:
                random.shuffle(txts_lc)
            txt_new = self.match_case(txts_lc[0], txt)
            return txt_new, self.rep_type
        return None, None

    def match_case(self, txt, txt_as):
        if txt_as.isupper():
            return txt.upper()
        if txt_as.islower():
            return txt
        if txt_as[0].isupper() and txt_as[1:].islower():
            return txt[0].upper() + txt[1:]
    
class Noise():
    def __init__(self, args):
        self.args = args
        self.grammar = Replacement(self.args.grammar, 'grammar') if self.args.grammar is not None else None
        self.homophone = Replacement(self.args.homophone, 'homophone') if self.args.homophone is not None else None
        self.misspell = Misspell(args, min_word_length=2)
        self.case = Case()
        self.hyphen = Hyphen()
        self.space = Space()
        self.duplicate = Duplicate()
        self.onmttok = pyonmttok.Tokenizer("aggressive", joiner_annotate=True, joiner=ONMTTOK_JOINER)
        
    def __call__(self, txt_clean): #noise the sentence txt_clean: "Bruxelles, 79 rue Montoyer."
        tok_clean = self.onmttok(txt_clean) #['Bruxelles', '￭,', '79', 'rue', 'Montoyer', '￭.']
        min_noises = int(len(tok_clean)*self.args.min_r)
        max_noises = int(len(tok_clean)*self.args.max_r)
        n_noises = random.randint(min_noises, max_noises) #number of noises to inject in current sentence, [min_noises, max_noises]
        idxs = [idx for idx in range(len(tok_clean))]
        random.shuffle(idxs)
        n_errors = 0
        self.tok_noisy = copy.deepcopy(tok_clean) ### this will contain initially tok_clean tokens and at the end some noisy tokens
        self.errors = ['-'] * len(tok_clean) ### initially no errors in tok_noisy
        for idx in idxs:
            if n_errors == n_noises:
                break
            tok, error = self.noise_tok(idx)
            if error is None:
                continue
            n_errors += 1
            self.tok_noisy[idx] = tok
            self.errors[idx] = error

        detok_noisy = ' '.join(self.tok_noisy) #"Bruxelles ￭, 79 rue Montoyer ￭."
        return tok_clean, self.tok_noisy, self.errors, self.onmttok.detokenize(detok_noisy.split())


    def del_joiners(self, txt):
        starts_with_joiner, ends_with_joiner = False, False
        if txt.startswith(ONMTTOK_JOINER):
            txt = txt[len(ONMTTOK_JOINER):]
            starts_with_joiner = True
        if txt.endswith(ONMTTOK_JOINER):
            txt = txt[:-len(ONMTTOK_JOINER)]
            ends_with_joiner = True
        return txt, starts_with_joiner, ends_with_joiner

    
    def add_joiners(self, txt, starts_with_joiner, ends_with_joiner):
        if starts_with_joiner:
            txt = ONMTTOK_JOINER + txt
        if ends_with_joiner:
            txt = txt + ONMTTOK_JOINER
        return txt

    
    def noise_tok(self,idx):
        curr_txt_nojoiners = copy.deepcopy(self.tok_noisy[idx]) #tok_noisy[idx] is NOT noisy
        curr_txt_nojoiners, starts_with_joiner, ends_with_joiner = self.del_joiners(curr_txt_nojoiners)
        if curr_txt_nojoiners.isnumeric():
            return None, None

        prev_tok = self.tok_noisy[idx-1] if idx > 0 else None #may be noisy
        next_tok = self.tok_noisy[idx+1] if idx < len(self.tok_noisy)-1 else None #may be noisy
        prev_is_noisy = self.errors[idx-1] != '-' if idx > 0 else None
        next_is_noisy = self.errors[idx+1] != '-' if idx < len(self.tok_noisy)-1 else None 
        for next_error in self.get_error_order(): # try all errors in get_error_order until one is injected
            if next_error == 'grammar':
                txt_noisy, noise = self.grammar(curr_txt_nojoiners)
            elif next_error == 'homophone':
                txt_noisy, noise = self.homophone(curr_txt_nojoiners)
            elif next_error == 'misspell':
                txt_noisy, noise = self.misspell(curr_txt_nojoiners)
            elif next_error == 'case':
                txt_noisy, noise = self.case(curr_txt_nojoiners)
            elif next_error == 'hyphen':
                txt_noisy, noise = self.hyphen(self.tok_noisy[idx], prev_tok, next_tok, prev_is_noisy, next_is_noisy)
                return txt_noisy, noise ### no need to add joiners
            elif next_error == 'space':
                txt_noisy, noise = self.space(curr_txt_nojoiners, prev_tok, next_tok)
            elif next_error == 'duplicate':
                txt_noisy, noise = self.duplicate(curr_txt_nojoiners)
                
            if noise is not None:
                return self.add_joiners(txt_noisy, starts_with_joiner, ends_with_joiner), noise

        return None, None


    def get_error_order(self):
        ### set the order of the sequence of errors according to the given weights
        errors2weights = {}
        if self.args.w_misspell and self.misspell is not None:
            errors2weights['misspell'] = self.args.w_misspell
            
        if self.args.w_homophone and self.homophone is not None:
            errors2weights['homophone'] = self.args.w_homophone
            
        if self.args.w_grammar and self.grammar is not None:
            errors2weights['grammar'] = self.args.w_grammar
            
        if self.args.w_case and self.case is not None:
            errors2weights['case'] = self.args.w_case
            
        if self.args.w_hyphen:
            errors2weights['hyphen'] = self.args.w_hyphen
            
        if self.args.w_space:
            errors2weights['space'] = self.args.w_space
            
        if self.args.w_duplicate:
            errors2weights['duplicate'] = self.args.w_duplicate
            
        next_errors = []
        for _ in range(len(errors2weights)):
            error = random.choices(list(errors2weights.keys()), list(errors2weights.values()), k=1)[0]
            next_errors.append(error)
            del errors2weights[error]
        return next_errors

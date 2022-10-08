import re
import sys
import random
import logging
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
    def __init__(self,wdelete=1,wrepeat=1,wreplace=1,wswap=1,wdiacritics=10,wdcons=15):
        self.wdelete = wdelete
        self.wrepeat = wrepeat
        self.wreplace = wreplace
        self.wswap = wswap
        self.wdiacritics = wdiacritics
        self.wdcons = wdcons
        logging.info('Built Misspell noiser')
        
    def __call__(self, word):
        words = []
        weights = []
        
        ### delete char i
        if len(word) > 1:
            for i in range(len(word)):
                weights.append(self.wdelete)
                words.append(word)
                words[-1] = words[-1][:i] + words[-1][i+1:]
                
        ### repeat char i
        for i in range(len(word)):
            weights.append(self.wrepeat)
            words.append(word)
            words[-1] = words[-1][:i] + words [-1][i] + words[-1][i:]
            
        ### swap chars i <=> i+1
        if len(word) > 1:
            for i in range(len(word)-1):
                c1 = word[i]
                c2 = word[i+1]
                weights.append(self.wswap)
                words.append(word)
                words[-1] = words[-1][:i] + c2 + c1 + words[-1][i+2:]

        ### replace char i by close to it
        for i in range(len(word)):
            near_letter = change_keyboard(word[i])
            if near_letter is not None:
                weights.append(self.wreplace)
                words.append(word)
                words[-1] = words[-1][:i] + near_letter + words[-1][i+1:]

        ### replace char i if vowel by the same with diacritics
        for i in range(len(word)):
            new_letter = change_accent(word[i])
            if new_letter is not None:
                weights.append(self.wdiacritics)
                words.append(word)
                words[-1] = words[-1][:i] + new_letter + words[-1][i+1:]

        ### delete/add char i if doubled consonant
        if len(word) >= 4:
            for i in range(1,len(word)-2): #delete a doubled consonne
                if word[i] == word[i+1] and word[i] in CONSONNES_DOUBLES:
                    weights.append(self.wdcons)
                    words.append(word)
                    words[-1] = words[-1][:i] + words[-1][i+1:] #char i is not added

            for i in range(1,len(word)-2): #add a doubled consonne
                if word[i] != word[i+1] and word[i] != word[i-1] and word[i] in CONSONNES_DOUBLES:
                    weights.append(self.wdcons)
                    words.append(word)
                    words[-1] = words[-1][:i+1] + words[-1][i:] #char i is added twice

        if len(words) == 0:
            return None
        
        return random.choices(words,weights)[0]

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
                return txt[0] + txt[1:].lower() #Ibm
            else:
                return txt.lower() # ibm
        
        if txt.islower(): #maison
            if len(txt) > 1 and random.random() < 0.5:
                return txt[0].upper() + txt[1:] #Maison
            else:
                return txt.upper() #MAISON
        
        if txt[0].isupper() and txt[1:].islower(): #Table
            if random.random() < 0.5:
                return txt.upper() #TABLE
            else:
                return txt.lower() #table
        return None

class Hyphen():
    def __init__(self):
        logging.info('Built Hyphen noiser')

    def __call__(self, txt_clean, prev_tok, post_tok): #txt may contain joiners
        if txt_clean != ONMTTOK_JOINER + '-' + ONMTTOK_JOINER:
            return None

        if not PATTERN_WORD.match(prev_tok) or not PATTERN_WORD.match(post_tok):
            return None

        if random.random() < 0.5:
            return ONMTTOK_JOINER #joins prev/post tokens
        
        return '' #deletes joiner
    
class Space():
    def __init__(self):
        logging.info('Built Space noiser')

    def __call__(self, txt_clean, prev_tok, post_tok): #txt_clean may contain joiners

        if not PATTERN_WORD.match(txt_clean):
            return None
        
        if random.random() < 0.5: ### split
            minlen = 3
            if len(txt_clean) < 2*minlen: #minimum length of resulting spltted tokens
                return None
            k = random.randint(minlen,len(txt_clean)-minlen)
            prev = txt_clean[:k]
            post = txt_clean[k:]
            return prev + " " + post

        else: ### merge
            ### merge with next
            if post_tok is not None and not post_tok.startswith(ONMTTOK_JOINER):
                return txt_clean + ONMTTOK_JOINER
            ### merge with prev
            if prev_tok is not None and not prev_tok.endswith(ONMTTOK_JOINER):
                return ONMTTOK_JOINER + txt_clean        
        return None

class Duplicate():
    def __init__(self):
        logging.info('Built Duplicate noiser')

    def __call__(self, txt_clean, prev_tok, post_tok): #txt_clean may contain joiners
        if not PATTERN_WORD.match(txt_clean):
            return None
        if prev_tok is not None and prev_tok.endswith(ONMTTOK_JOINER):
            return None
        if post_tok is not None and post_tok.startswith(ONMTTOK_JOINER):
            return None
        return txt_clean + ' ' + txt_clean
    
class Replacements():
    def __init__(self,f,min_error=2):
        self.replacements = defaultdict(list)
        self.min_error = min_error ### min error length
        with open(f,'r') as fd:
            for l in fd:
                toks = l.rstrip().split('\t')
                self.replacements[toks[0]] = toks[1:]
        logging.info('Loaded Replacements file with {} entries'.format(f,len(self.replacements)))
                
    def __call__(self,txt):
        if len(txt) < self.min_error: 
            return None 
        txt_lc = txt.lower()
        if txt_lc in self.replacements:
            txts_lc = self.replacements[txt_lc]
            if len(txts_lc) > 1:
                random.shuffle(txts_lc)
            txt_new = self.match_case(txts_lc[0], txt)
            return txt_new
        return None

    def match_case(self, txt, txt_as):
        if txt_as.isupper():
            return txt.upper()
        if txt_as.islower():
            return txt
        if txt_as[0].isupper() and txt_as[1:].islower():
            return txt[0].upper() + txt[1:]
    

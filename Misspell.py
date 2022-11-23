import re
import copy
import random
import logging
import unicodedata
from collections import defaultdict

CONSONNES_DOUBLES = 'dcflsptmnrgDCFLSPTMNRG' #bkz
ACCENTS = 'aaàáâäeeéèêëiiíìîïooóòôöuuúùûüAAÀÁÂÄEEÉÈÊËIIÍÌÎÏOOÓÒÔÖUUÚÙÛÜ'
ACCENTS_FREQ = { 'a': {'a':20, 'á':1,  'à':10, 'â':1, 'ä':0},
                 'e': {'e':20, 'é':10, 'è':9,  'ê':2, 'ë':1},
                 'i': {'i':20, 'í':3,  'ì':0,  'î':1, 'ï':1},
                 'o': {'o':20, 'ó':0,  'ò':0,  'ô':2, 'ö':1},
                 'u': {'u':20, 'ú':0,  'ù':2,  'û':1, 'ü':1}}

KEYBOARD_PUNCT = {'-': '0=p[oòóôö]',
                  '=': '-[]p\\',
                  '[': "p]'-=;0l",
                  ']': "[\\='-;",
                  '\\': "]='",
                  ';': "l'/p[.oòóôö],",
                  "'": ';[]/p\\.',
                  ',': 'm.klj;',
                  '.': ",/l;k'",
                  '/': ".;'l",
                  '~': '!Q',
                  '!': '~@QW',
                  '@': '!#QWEÈÉÊË',
                  '#': '@$WEÈÉÊËQR',
                  '$': '#%EÈÉÊËRWT',
                  '%': '$^RTEÈÉÊËY',
                  '^': '%&TYRUÙÚÛÜ',
                  '&': '^*YUÙÚÛÜTIÌÍÎÏ',
                  '*': '&(UÙÚÛÜIÌÍÎÏYOÒÓÔÖ',
                  '(': '*)IÌÍÎÏOÒÓÔÖUÙÚÛÜP',
                  ')': '(_OÒÓÔÖPIÌÍÎÏ{',
                  '_': ')+P{OÒÓÔÖ}',
                  '+': '_{}P|',
                  '{': 'P}"_+:)L',
                  '}': '{|+"_:',
                  '|': '}+"',
                  ':': 'L"?P{>OÒÓÔÖ}<',
                  '"': ':{}?P|>',
                  '<': 'M>KLJ:',
                  '>': '<?L:K"',
                  '?': '>:"L'}

KEYBOARD_DIGITS = {'1': '`2qw',
                   '2': '13qweèéêë',
                   '3': '24weèéêëqr',
                   '4': '35eèéêërwt',
                   '5': '46rteèéêëy',
                   '6': '57tyruùúûü',
                   '7': '68yuùúûütiìíîï',
                   '8': '79uùúûüiìíîïyoòóôö',
                   '9': '80iìíîïoòóôöuùúûüp',
                   '0': '9-oòóôöpiìíîï['}

KEYBOARD_LETTER_orig = {'q': 'waàáâä12s`3',
                   'w': 'qeèéêës23aàáâäd14',
                   'e': 'èéêëwrd34sf25aàáâä',
                   'r': 'eèéêëtf45dg36s',
                   't': 'ryg56fh47d',
                   'y': 'tuùúûüh67gj58f',
                   'u': 'ùúûüyiìíîïj78hk69g',
                   'i': 'ìíîïuùúûüoòóôök89jl7',
                   'o': 'òóôöiìíîïpl90k;8-j',
                   'p': "oòóôö[;0-l'9=k",
                   'a': 'àáâäszqwxeèéêë',
                   's': 'aàáâädxweèéêëzcçqrf',
                   'd': 'fscçeèéêërxvwtzg',
                   'f': 'dgvrtcçbeèéêëyxs',
                   'g': 'fhbtyvnruùúûücçd',
                   'h': 'gjnyuùúûübmtiìíîïv',
                   'j': 'hkmuùúûüiìíîïn,yoòóô',
                   'k': 'jl,iìíîïoòóôöm.uùúûü',
                   'l': 'k;.oòóôöp,/iìíîï[m',
                   'z': 'xaàáâäsd',
                   'x': 'zcçsdaàáâäf',
                   'c': 'çvxdfsg',
                   'v': 'cçbfgdh',
                   'b': 'vnghfj',
                   'n': 'bmhjgk',
                   'm': 'n,jkhl',
                   'Q': 'WAÀÁÂÄ!@S~#',
                   'W': 'QEÈÉÊËS@#AÀÁÂÄD!$',
                   'E': 'ÈÉÊËWRD#$SF@%AÀÁÂÄ',
                   'R': 'EÈÉÊËTF$%DG#^S',
                   'T': 'RYG%^FH$&D',
                   'Y': 'TUÙÚÛÜH^&GJ%*F',
                   'U': 'ÙÚÛÜYIÌÍÎÏJ&*HK^(G',
                   'I': 'ÌÍÎÏUÙÚÛÜOÒÓÔÖK*(JL&',
                   'O': 'ÒÓÔÖIÌÍÎÏPL()K:*_J',
                   'P': 'OÒÓÔÖ{:)_L"(+K',
                   'A': 'ÀÁÂÄSZQWXEÈÉÊË',
                   'S': 'AÀÁÂÄDXWEÈÉÊËZCÇQRF',
                   'D': 'FSCÇEÈÉÊËRXVWTZG',
                   'F': 'DGVRTCÇBEÈÉÊËYXS',
                   'G': 'FHBTYVNRUÙÚÛÜCÇD',
                   'H': 'GJNYUÙÚÛÜBMTIÌÍÎÏV',
                   'J': 'HKMUÙÚÛÜIÌÍÎÏN<YOÒÓÔ',
                   'K': 'JL<IÌÍÎÏOÒÓÔÖM>UÙÚÛÜ',
                   'L': 'K:>OÒÓÔÖP<?IÌÍÎÏ{M',
                   'Z': 'XAÀÁÂÄSD',
                   'X': 'ZCÇSDAÀÁÂÄF',
                   'C': 'ÇVXDFSG',
                   'V': 'CÇBFGDH',
                   'B': 'VNGHFJ',
                   'N': 'BMHJGK',
                   'M': 'N<JKHL'}

KEYBOARD_LETTER = {'q': 'waàáâäs',
                   'w': 'qeèéêësaàáâäd',
                   'e': 'èéêëwrdsfaàáâä',
                   'r': 'eèéêëtfdgs',
                   't': 'rygfhd',
                   'y': 'tuùúûühgjf',
                   'u': 'ùúûüyiìíîïjhkg',
                   'i': 'ìíîïuùúûüoòóôökjl',
                   'o': 'òóôöiìíîïplkj',
                   'p': "oòóôölk",
                   'a': 'àáâäszqwxeèéêë',
                   's': 'aàáâädxweèéêëzcçqrf',
                   'd': 'fscçeèéêërxvwtzg',
                   'f': 'dgvrtcçbeèéêëyxs',
                   'g': 'fhbtyvnruùúûücçd',
                   'h': 'gjnyuùúûübmtiìíîïv',
                   'j': 'hkmuùúûüiìíîïnyoòóô',
                   'k': 'jliìíîïoòóôömuùúûü',
                   'l': 'koòóôöpiìíîïm',
                   'z': 'xaàáâäsd',
                   'x': 'zcçsdaàáâäf',
                   'c': 'çvxdfsg',
                   'v': 'cçbfgdh',
                   'b': 'vnghfj',
                   'n': 'bmhjgk',
                   'm': 'njkhl',
                   'Q': 'WAÀÁÂÄS',
                   'W': 'QEÈÉÊËS@#AÀÁÂÄD',
                   'E': 'ÈÉÊËWRDSFAÀÁÂÄ',
                   'R': 'EÈÉÊËTFDGS',
                   'T': 'RYGFHD',
                   'Y': 'TUÙÚÛÜHGJF',
                   'U': 'ÙÚÛÜYIÌÍÎÏJHKG',
                   'I': 'ÌÍÎÏUÙÚÛÜOÒÓÔÖKJL',
                   'O': 'ÒÓÔÖIÌÍÎÏPLKJ',
                   'P': 'OÒÓÔÖLK',
                   'A': 'ÀÁÂÄSZQWXEÈÉÊË',
                   'S': 'AÀÁÂÄDXWEÈÉÊËZCÇQRF',
                   'D': 'FSCÇEÈÉÊËRXVWTZG',
                   'F': 'DGVRTCÇBEÈÉÊËYXS',
                   'G': 'FHBTYVNRUÙÚÛÜCÇD',
                   'H': 'GJNYUÙÚÛÜBMTIÌÍÎÏV',
                   'J': 'HKMUÙÚÛÜIÌÍÎÏNYOÒÓÔ',
                   'K': 'JLIÌÍÎÏOÒÓÔÖMUÙÚÛÜ',
                   'L': 'KOÒÓÔÖPIÌÍÎÏM',
                   'Z': 'XAÀÁÂÄSD',
                   'X': 'ZCÇSDAÀÁÂÄF',
                   'C': 'ÇVXDFSG',
                   'V': 'CÇBFGDH',
                   'B': 'VNGHFJ',
                   'N': 'BMHJGK',
                   'M': 'NJKHL'}

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
    if letter not in KEYBOARD_LETTER: # and letter not in KEYBOARD_PUNCT and letter not in KEYBOARD_DIGITS:
        return None
    if letter in KEYBOARD_LETTER:
        near_letters = list(KEYBOARD_LETTER[letter])
    #elif letter in KEYBOARD_DIGITS:
    #    near_letters = list(KEYBOARD_DIGITS[letter])
    #elif letter in KEYBOARD_PUNCT:
    #    near_letters = list(KEYBOARD_PUNCT[letter])
        
    if letter in near_letters:
        near_letters.remove(letter)
    random.shuffle(near_letters)
    return near_letters[0]    

class Misspell():

    def __init__(self,opts, name):
        self.opts = opts
        self.name = name
        self.min_word_length = 2 #do not misspell if word is too small
        self.n_misspell = defaultdict(int)
        self.n = 0
        
    def __call__(self, word):
        words, types, weights = [], [], []
        if len(word) < self.min_word_length:
            return '', ''
        
        for i in range(len(word)):
            
            ### delete char i
            if len(word) > 1:
                weights.append(self.opts.delete)
                types.append('MISSPELL:delete')
                words.append(word)
                words[-1] = words[-1][:i] + words[-1][i+1:]

            ### repeat char i
            weights.append(self.opts.repeat)
            types.append('MISSPELL:repeat')
            words.append(word)
            words[-1] = words[-1][:i+1] + words[-1][i:]

            ### swap chars i <=> i+1
            if i < len(word)-1:
                c1 = word[i]
                c2 = word[i+1]
                weights.append(self.opts.swap)
                types.append('MISSPELL:swap')
                words.append(word)
                words[-1] = words[-1][:i] + c2 + c1 + words[-1][i+2:]

            ### replace char i by close to it
            near_letter = change_keyboard(word[i])
            if near_letter is not None:
                weights.append(self.opts.close)
                types.append('MISSPELL:close')
                words.append(word)
                words[-1] = words[-1][:i] + near_letter + words[-1][i+1:]

            ### replace char i if vowel by the same with diacritics
            new_letter = change_accent(word[i])
            if new_letter is not None:
                weights.append(self.opts.diacritics)
                types.append('MISSPELL:diacr')
                words.append(word)
                words[-1] = words[-1][:i] + new_letter + words[-1][i+1:]

            ### delete/add char i if doubled consonant
            if i>0 and i<len(word)-2 and word[i] == word[i+1] and word[i] in CONSONNES_DOUBLES:
                weights.append(self.opts.consd)
                types.append('MISSPELL:consd:del')
                words.append(word)
                words[-1] = words[-1][:i] + words[-1][i+1:] #char i is not added

            ### add double consonnant
            if i>0 and i<len(word)-1 and word[i] != word[i+1] and word[i] != word[i-1] and word[i] in CONSONNES_DOUBLES:
                weights.append(self.opts.consd)
                types.append('MISSPELL:consd:add')
                words.append(word)
                words[-1] = words[-1][:i+1] + words[-1][i:] #char i is added twice

            ### 'ph' -> 'f'
            if len(word)>i+1 and word[i] == 'p' and word[i+1] == 'h':
                weights.append(self.opts.phone)
                types.append('MISSPELL:phone:ph2f')
                words.append(word)
                words[-1] = words[-1][:i] + 'f' + words[-1][i+2:]
                
            ### 'h' -> 
            if word[i] == 'h' and not (i>0 and word[i-1] in 'CcPp'):
                weights.append(self.opts.phone)
                types.append('MISSPELL:phone:h2-')
                words.append(word)
                words[-1] = words[-1][:i] + words[-1][i+1:]

            ### '.+c[EI]' -> '.+ss[EI]' #vacances -> vacansses
            if i>0 and len(word)>i+1 and word[i] == 'c' and word[i+1].lower() in 'eèéi':
                weights.append(self.opts.phone)
                types.append('MISSPELL:phone:c2ss')
                words.append(word)
                words[-1] = words[-1][:i] + 'ss' + words[-1][i+1:]

            ### 'ç' -> 'ss' #leçon -> lesson
            if word[i] == 'ç':
                weights.append(self.opts.phone)
                types.append('MISSPELL:phone:ç2ss')
                words.append(word)
                words[-1] = words[-1][:i] + 'ss' + words[-1][i+1:]
                
        if len(words) == 0:
            return '', ''

        ### select noisy to inject
        i = random.choices([i for i in range(len(words))],weights)[0]
        self.n += 1
        self.n_misspell[types[i]] += 1
        return words[i], self.name #types[i]

    def stats(self):
        if self.n:
            for k, v in sorted(self.n_misspell.items(), key=lambda item: item[1], reverse=True): #if reverse, sorted in descending order
                logging.info('{}\t{:.2f}%\t{}'.format(v,100.0*v/self.n,k))

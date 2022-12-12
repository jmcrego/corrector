# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch
import time
from mosestokenizer import MosesDetokenizer
#from GECor import save_checkpoint
from collections import defaultdict
from utils.Utils import SEPAR1, SEPAR2, KEEP

minp_cor = 0.1

class Inference():

    def __init__(self, model, testset, tags, cors, token, lex, args, device):
        self.detok = MosesDetokenizer('fr')
        super(Inference, self).__init__()
        softmax = torch.nn.Softmax(dim=-1)
        self.args = args
        self.token = token
        self.tags = tags
        self.cors = cors #either Vocabulary or None
        self.lex = lex
        idx_PAD_tag = tags.idx_PAD
        idx_PAD_cor = cors.idx_PAD if cors is not None else 2 #<pad> in FlaubertTok
        self.corrected_testset = [None] * len(testset) #final string to output of corrected sentences
        dinputs = {}
        model.eval()
        with torch.no_grad():
            for inputs, indexs, _, _, idxs, words in testset:
                logging.info(idxs)
                (bs, l) = inputs.shape
                #inputs [[ 0, 147, 3291, 49, 13578, 37, 246, 5067, 530, 61, 1, 2]] => [bs,l]
                #indexs [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 10, 11, 12]] => [bs,l]
                #words  [['<s>', 'Un', 'sourire', 'se', 'dessine', 'sur', 'mes', 'lèvres', 'lorsque', 'je', 'revois','</s>']]
                #for i in range(bs):
                #    logging.debug('idx={}\n\t{}\n\t{}\n\t{}'.format(idxs[i],inputs[i].tolist(),indexs[i].tolist(),words[i]))
                for n_iter in range(args.max_iter):
                    logging.debug('n_iter:{}'.format(n_iter))
                    inputs = inputs.to(device)
                    indexs = indexs.to(device)
                    dinputs['input_ids'] = inputs
                    outtag, outcor = model(dinputs, indexs) ### [bs, l, ts], [bs, l, ws] or [bs, l, nsubt*ws]
                    #logging.info('outtag.shape = {}'.format(outtag.shape))
                    #logging.info('outcor.shape = {}'.format(outcor.shape))
                    #sorted_tags = outtag.argsort(dim=2, descending=True)[:,:,:args.Kt] #[bs, l, Kt]
                    #sorted_cors = outcor.argsort(dim=2, descending=True)[:,:,:args.Kc] #[bs, l, Kc]

                    ### TAGS ###
                    outtag = softmax(outtag) #[bs, l, ts]
                    sorted_tags_prob, sorted_tags = torch.sort(outtag, dim=-1, descending=True)
                    #sorted_tags_prob = sorted_tags_prob[:,:,:args.Kt]
                    #sorted_tags = sorted_tags[:,:,:args.Kt]

                    ### CORS ###
                    if args.n_subtokens > 1:
                        bs, l, nsubt_times_ws = outcor.shape
                        outcor = outcor.reshape(bs,l,args.n_subtokens,-1) #[bs, l, n_subt, ws]
                    outcor = softmax(outcor) #[bs, l, n_subt, ws]
                    sorted_cors_prob, sorted_cors = torch.sort(outcor, dim=-1, descending=True)
                    #sorted_cors_prob = sorted_cors_prob[:,:,:args.Kc]
                    #sorted_cors = sorted_cors[:,:,:args.Kc]

                    continue_batch, continue_idx = self.correct_batch(idxs,words,sorted_tags,sorted_cors,sorted_tags_prob,sorted_cors_prob)
                    if len(continue_batch) == 0:
                        break
                    inputs, indexs, words = testset.reformat_batch(continue_batch) ### prepare for next iteration
                    idxs =  continue_idx
                
        for i in range(len(self.corrected_testset)):
            #print(' '.join(self.corrected_testset[i][1:-1]))
            #print(self.detok(self.corrected_testset[i][1:-1]))
            print(self.corrected_testset[i])


    def correct_batch(self,idxs,words,sorted_tags,sorted_cors,sorted_tags_prob,sorted_cors_prob):
        continue_batch = []
        continue_idxs = []
        for s in range(sorted_tags.shape[0]): #sentence by sentence
            sorted_tags_s, sorted_tags_prob_s = self.tags_idsToStr(sorted_tags[s], sorted_tags_prob[s])
            sorted_cors_s, sorted_cors_prob_s = self.cors_idsToStr(sorted_cors[s], sorted_cors_prob[s])
            corrected_sentence, is_corrected = self.correct_sentence(words[s],sorted_tags_s,sorted_cors_s,sorted_tags_prob_s,sorted_cors_prob_s,idxs[s])
            self.corrected_testset[idxs[s]] = self.detok(corrected_sentence)            
            if is_corrected: ### more corrections needed
                continue_batch.append(self.corrected_testset[idxs[s]])
                continue_idxs.append(idxs[s])
        return continue_batch, continue_idxs

    def tags_idsToStr(self, sorted_tags, sorted_tags_prob):
        tags = []
        tags_prob = []
        for w in range(len(sorted_tags)): #<s> and </s> are not needed
            curr_tags = [self.tags[tag_idx] for tag_idx in sorted_tags[w,:self.args.Kt].tolist()]
            curr_tags_prob = [tag_prb for tag_prb in sorted_tags_prob[w,:self.args.Kt].tolist()]
            tags.append(curr_tags)
            tags_prob.append(curr_tags_prob)
        return tags, tags_prob
            
    def cors_idsToStr(self, sorted_cors, sorted_cors_prob):
        #sorted_cors [l, n_subt, Vw]
        #sorted_cors_prob [l, n_subt, Vw]
        cors = []
        cors_prob = []
        for w in range(len(sorted_cors)): #<s> and </s> are not needed
            if self.args.n_subtokens > 1:
                curr_cors = [self.token.get_subtoks_joined(sorted_cors[w,:,0].tolist())] #take one best of each subtoken
                curr_cors_prob = [torch.prod(sorted_cors_prob[w,:,0]).tolist()]
            else:
                curr_cors = [self.cors[cor_idx] for cor_idx in sorted_cors[w,:self.args.Kc].tolist()]
                curr_cors_prob = [cor_prb for cor_prb in sorted_cors_prob[w,:self.args.Kc].tolist()]
            cors.append(curr_cors)
            cors_prob.append(curr_cors_prob)
        return cors, cors_prob
    
    def correct_sentence(self,words,sorted_tags,sorted_cors,sorted_tags_prob,sorted_cors_prob,idx):
        #words            [l]
        #sorted_tags      [l,Kt]
        #sorted_tags_prob [l,Kt]
        #sorted_cors      [l,Kc]
        #sorted_cors_prob [l,Kc]
        words_corrected = [] #['<s>']
        is_corrected = False
        logging.debug('IN[{}] : {}'.format(idx,words))
        for i in range(1,len(words)-1): ### do not correct <s> and </s>
            if words[i] == "": ### has been deleted
                continue
            logging.debug("\t{}\t{}\t{}\t{}\t{}".format(words[i], sorted_tags[i], sorted_cors[i], sorted_tags_prob[i], sorted_cors_prob[i]))
            corrected_word, delete_next = self.correct_word(words[i], sorted_tags[i], sorted_cors[i], sorted_tags_prob[i], sorted_cors_prob[i], next_word=words[i+1])
            if corrected_word != words[i] or delete_next:
                is_corrected = True
            logging.debug("\t\t{}\tdelete_next:{}".format(corrected_word, delete_next))
            words_corrected.append(corrected_word)
            #words[i] = corrected_word
            if delete_next and i<len(words)-1:
                words[i+1] = ""
        #words_corrected.append('</s>')
        logging.debug('OUT[{}]: {}'.format(idx,words_corrected))
        return words_corrected, is_corrected

    def correct_word(self, word, curr_tags, curr_cors, curr_tags_prob, curr_cors_prob, next_word=None):

        for k in range(len(curr_tags)):
            if curr_tags[k] == KEEP:
                return word, False #no correction
        
            elif curr_tags[k] == '$LEMM':
                i = self.find_samelemma(word,curr_cors,curr_tags_prob[k],curr_cors_prob)
                if i>=0:
                    return curr_cors[i], False
                continue #try next tag
        
#            elif curr_tags[k] == '$POS':
#                i = self.find_samepos(word,curr_cors,curr_tags_prob[k],curr_cors_prob)
#                if i>=0:
#                    return curr_cors[i], False
#                continue #try next tag
        
            elif curr_tags[k] == '$PHON':
                i = self.find_homophone(word,curr_cors,curr_tags_prob[k],curr_cors_prob)
                if i>=0:
                    return curr_cors[i], False
                continue #try next tag
        
            elif curr_tags[k] == '$SPEL':
                i = self.find_spell(word,curr_cors,curr_tags_prob[k],curr_cors_prob)
                if i>=0:
                    return curr_cors[i], False
                continue #try next tag

            elif curr_tags[k] == '$APND':
                return word + " " + curr_cors[0], False
        
            elif curr_tags[k] == '$DELE':
                return "", False
        
            elif curr_tags[k] == '$MRGE':
                if next_word != '</s>':
                    if word+next_word in self.lex:
                        return word + next_word, True ### must delete next word
                continue #try next tag
            
            elif curr_tags[k] == '$SWAP':
                if next_word != '</s>':
                    return next_word + " " + word, True ### must delete next word
                continue #try next tag
            
            elif curr_tags[k] == '$SPLT':
                i = self.find_split(word,curr_cors,curr_tags_prob[k],curr_cors_prob)
                if i>=0:
                    w1 = curr_cors[i]
                    l = len(w1)
                    w2 = word[l+1:]
                    if w2[0] == '-':
                       w2 = w2[1:]
                    return w1 + " " + w2, False
                continue #try next tag
            
            elif curr_tags[k] == '$CAS1':
                if not word[0].isalpha():
                    continue                
                if word[0].isupper():
                    word = word[0].lower() + word[1:]
                elif word[0].islower():
                    word = word[0].upper() + word[1:]
                return word, False
        
            elif curr_tags[k] == '$CASn':
                if word.islower():
                    return word.upper(), False
                elif word.isupper():
                    return word.lower(), False
                continue #try next tag
            
            elif curr_tags[k] == '$HYPs':
                i = word.find('-')
                if i >= 0:
                    w1 = word[:i]
                    w2 = word[i+1:]
                    return w1 + " " + w2, False
                continue #return word, False ### no change

            elif curr_tags[k] == '$HYPm':
                if next_word != '</s>':
                    #if word+ "-" + next_word in self.lex:
                    return word + "-" + next_word, True ### must delete next word
                continue #return word, False ### no change
            
            #elif curr_tags[k].startswith('$INFLECT:'):
            #    newword = self.inflect(curr_tags[k][9:],word)
            #    if len(newword) > 0:
            #        if self.in_lexicon(newword):
            #            return newword, False
            #    continue #return word, False ### no change
        
            else:
                logging.error('Bad tag: '.format(curr_tags[k]))
                sys.exit()

        return word, False ### no change


    def lowercase_first(self, txt):
        first_is_upper = True if txt[0].isupper() else False
        if first_is_upper:
            txt = txt[0].upper() + txt[1:]
        return first_is_upper, txt

    
    def find_split(self, word, curr_cors, tag_prob, curr_cors_prob):
        logging.debug('\tfind_split({})'.format(word))
        for i, cor in enumerate(curr_cors):
            if curr_cors_prob[i] < minp_cor:
                return -1
            l = len(cor)
            w1 = curr_cors[i]
            if len(w1) >= len(word):
                continue
            if word[:len(w1)] != w1:
                continue
            w2 = word[l+1:]
            if len(w2) == 0:
                continue
            if w2[0] == '-':
                w2 = w2[1:]
            if not w1 in self.lex or not w2 in self.lex:
                continue
            logging.debug('\t\tfirst split: {}'.format(curr_cors[i]))
            return i
        return -1

    def find_samelemma(self, word, curr_cors, tag_prob, curr_cors_prob):
        logging.debug('\tfind_samelemma({})'.format(word))
        first_is_upper_word, word = self.lowercase_first(word)
        lem_word = self.lex.txt2lem[word]
        for i, cor in enumerate(curr_cors):
            if curr_cors_prob[i] < minp_cor:
                return -1
            if cor == word:
                continue
            first_is_upper_cor, cor = self.lowercase_first(cor)
            lem_cor = self.lex.txt2lem[cor]
            if len(lem_cor.intersection(lem_word)) > 0:
                logging.debug('\t\tfirst samelem: {}'.format(curr_cors[i]))
                return i
        return -1
                      
    def find_samepos(self, word, curr_cors, tag_prob, curr_cors_prob):
        logging.debug('\tfind_samepos({})'.format(word))
        first_is_upper_word, word = self.lowercase_first(word)
        pos_word = self.lex.txt2pos[word]
        for i, cor in enumerate(curr_cors):
            if curr_cors_prob[i] < minp_cor:
                return -1
            if cor == word:
                continue
            first_is_upper_cor, cor = self.lowercase_first(cor)
            pos_cor = self.lex.txt2pos[cor]
            if len(pos_cor.intersection(pos_word)) > 0:
                logging.debug('\t\tfirst samepos: {}'.format(curr_cors[i]))
                return i
        return -1
    
    def find_homophone(self, word, curr_cors, tag_prob, curr_cors_prob):
        logging.debug('\tfind_homophone({})'.format(word))
        first_is_upper_word, word = self.lowercase_first(word)
        pho_word = self.lex.txt2pho[word]
        for i, cor in enumerate(curr_cors):
            if curr_cors_prob[i] < minp_cor:
                return -1
            if cor == word:
                continue
            first_is_upper_cor, cor = self.lowercase_first(cor)
            pho_cor = self.lex.txt2pho[cor]            
            if len(pho_cor.intersection(pho_word)) > 0:
                logging.debug('\t\tfirst samepos: {}'.format(curr_cors[i]))
                return i
        return -1
        

    def is_spell(self, wrd1, wrd2):
        swrd1 = set(list(wrd1))
        swrd2 = set(list(wrd2))
        logging.debug('swrd1: {}'.format(swrd1))
        logging.debug('swrd2: {}'.format(swrd2))
        if len(swrd1-swrd2) <= 1 and len(swrd2-swrd1) <= 1:
            return True
        return False

    
    def find_spell(self, word, curr_cors, tag_prob, curr_cors_prob):
        logging.debug('\tfind_spell({})'.format(word))
        for i, cor in enumerate(curr_cors):
            if curr_cors_prob[i] < minp_cor:
                return -1
            if cor == word:
                continue
            #if self.is_spell(word, cor):
            #    logging.debug('\tfirst spell: {}'.format(cor))
            #    return i
            logging.debug('\tfirst spell: {}'.format(cor))
            return i
        return -1
    
    
    def inflect(self, tag0, word):
        logging.debug('inflect({}, {})'.format(tag0,word))
        word_lc = word.lower()
        if word_lc not in self.lex.wrd2lem:
            logging.debug('word: {} not found in lex'.format(word_lc))
            return word
        
        feats = tag0.split(';')
        logging.debug('spacy feats: {}'.format(feats))
        pos = feats.pop(0)
        logging.debug('spacy pos: {}'.format(pos))
        lems = self.lex.wrd2lem[word_lc]
        logging.debug('lems: {}'.format(lems))

        for lem in lems:
            acceptable_words = self.lex.lempos2wrds[lem+separ+pos]
            logging.debug('acceptable_words: {}'.format(acceptable_words))
            for feat in feats:
                logging.debug('feat: {}'.format(feat))
                if lem+separ+pos+separ+feat in self.lex.lemposfeat2wrds:
                    reduced_words = self.lex.lemposfeat2wrds[lem+separ+pos+separ+feat]
                    logging.debug('feat: {} reduced_words: {}'.format(feat,reduced_words))
                    acceptable_words = acceptable_words.intersection(reduced_words)
                    logging.debug('feat: {} acceptable_words: {}'.format(feat,acceptable_words))
            if len(acceptable_words) == 1:
                return list(acceptable_words)[0]
        logging.debug('no inflection found')
        return 'inflect(' + word + '|' + tag0 + ')'


# -*- coding: utf-8 -*-
"""
Created on Tue May 30 17:21:39 2017

@author: Zheng Xin
"""


import numpy as np
import string
import pandas
import os
import gensim
from math import ceil
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
#import sys
#import csv
#dname = os.path.dirname(os.path.realpath(__file__))
#sys.path += [os.path.dirname(dname) + '/TicketCategorization']




class CharNumberEncoder(object):

    def __init__(self, data_iterator, word_len=30, sent_len=200):
        '''
        DESCRIPTIONS:
            This class converts text to numbers for the standard unicode vocabulary
            size.
        PARAMS:
            data_iterator (iterator): iterator to iterates the text strings
            word_len (int): maximum length of the word, any word of length less
                than that will be padded with zeros, any word of length more than
                that will be cut at max word length.
            sent_len (int): maximum number of words in a sentence, any sentence
                with less number of words than that will be padded with zeros,
                any sentence with more words than the max number will be cut at
                the max sentence length.
        '''
        self.data_iterator = data_iterator
        self.word_len = word_len
        self.sent_len = sent_len
        self.char_map = {}
        for i, ch in enumerate(string.printable):
            self.char_map[ch] = i+1 # hash character to number, leave 0 for blank space


    def make_char_embed(self):
        '''build array vectors of words and sentence, automatically skip non-ascii
           words.
        '''
        sents = []
        count = 0
        for paragraph in self.data_iterator:
            word_toks = paragraph.split(' ')
            word_vec = []
            for word in word_toks:
                word = word.strip()
                try:
                    word.encode('ascii')
                except:
                    #print '..Non ASCII Word', word
                    count += 1
                    continue
                if len(word) > 0:
                    word_vec.append(self.spawn_word_vec(word))

            if len(word_vec) > self.sent_len:
                sents.append(word_vec[:self.sent_len])
            else:
                zero_pad = np.zeros((self.sent_len-len(word_vec), self.word_len))
                if len(word_vec) > 0:
                    sents.append(np.vstack([np.asarray(word_vec), zero_pad]))
                else:
                    sents.append(zero_pad)
        print 'Non ASCII word number:\t', count
        return np.asarray(sents)


    def spawn_word_vec(self, word):
        '''Convert a word to number vector with max word length, skip non-ascii
           characters
        '''
        word_vec = []
        for c in word:
            try:
                assert c in self.char_map and c != ' ', '({}) of {} not in char map'.format(c,word)
            except:
                continue
            word_vec.append(self.char_map[c])
        if len(word_vec) > self.word_len:
            return word_vec[:self.word_len]
        else:
            word_vec += [0]*(self.word_len-len(word_vec))
        return word_vec



def onehot(X, nclass):
    encoder = OneHotEncoder(n_values=nclass)
    return encoder.fit_transform(X).toarray()
    
   
def data_char():  #num_train, word_len
    df1 = pandas.read_csv('./data/ag_train.csv')    
    y = df1[df1.columns[0]].values    
    X = df1[df1.columns[2]].values
    print len(X)
    print len(y)

    word_len = 20
    sent_len = 50
    num_train = 120000
    data = CharNumberEncoder(X, word_len=word_len, sent_len=sent_len)
    X_charIdx = data.make_char_embed() 
    a,b,c = X_charIdx.shape
    X_charIdx = np.reshape(X_charIdx, (a, b, c, 1))
    print 'text encoding done'
    
    train_X = X_charIdx[:num_train]
    valid_X = X_charIdx[num_train:]
    
    train_ys = []
    valid_ys = []
    y_idx = np.asarray(y) 
    nclass = 4
    components = [nclass+1]
    for l, n_comp in enumerate(components):
        y = y_idx
        y = onehot(y[:,np.newaxis], n_comp)
        print l, n_comp
        print 'y shape', y.shape
        train_ys = y[:num_train]
        valid_ys = y[num_train:]
        #train_ys.append(y[:num_train])
        #valid_ys.append(y[num_train:])
    train_ys = np.asarray(train_ys)
    valid_ys = np.asarray(valid_ys)
    print 'train_y shape', train_ys.shape
    print 'valid_y shape', valid_ys.shape
    
    return train_X, train_ys, valid_X, valid_ys
    #, components, sent_len, train_WX, valid_WX


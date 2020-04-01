import json
import codecs
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from time import time
from collections import defaultdict
from sklearn.manifold import TSNE
import multiprocessing
import gensim
from gensim.models import Word2Vec
# from gensim.models import Word2Vec
# from gensim.models.phrases import Phrases, Phraser
import logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

#data preprocessing
drg = clean_data('.\\drg\\drg-without-comma.json', '.\\drg\\DRG-klasifikace-XPath-vol9_nocodes.xlsx')

for entity in drg:
    entity['elements'] = [UDPipe_preprocessing_word(ls) for ls in entity['elements']]

for entity in drg:
    entity['elements'] = sentence_preprocessing(entity['elements'])# 1 908 documents
    
    #kapský, metr  celsius

with open('drg_clean_for_correction_json_reverseudpipe and sentence.txt', 'w', encoding = 'utf8') as outfile:
    json.dump(drg, outfile, ensure_ascii=False)         

with open('drg_clean_for_correction_json_reverseudpipe and sentence.txt', 'r', encoding = 'utf8') as file:
    drg_UDPipe = json.load(file)

for entity in drg:
    for i in range(len(entity['elements'])):# rucne zasiahnutie zle nalemmatizovanych slov .... treba regex na rozne tvary     
        entity['elements'][i] = re.sub(' moci ', ' nemoc ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' předloktit ', ' předloktí ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' nadloktit ', ' nadloktí ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' pažet ', ' paže ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' stehný ', ' stehno ', entity['elements'][i], flags=re.UNICODE)     ###
       

for entity in drg:
    entity['elements'] = " ".join(entity['elements'])

for entity in drg:
    entity['elements'] = list(entity['elements'].split())
        
    
with open('drg_clean_preprocessed_json_reversed.txt', 'w', encoding = 'utf8') as outfile:
    json.dump(drg, outfile, ensure_ascii=False)         

with open('drg_clean_preprocessed_json_reversed.txt', 'r', encoding = 'utf8') as file:
    drg_UDPipe = json.load(file)

del ls

##### word2vec
words = []
for entity in drg_UDPipe:
    words.append(entity['elements'])
    
word_freq = defaultdict(int)
for sent in words:
    for i in sent:
        word_freq[i] += 1
len(word_freq)

sorted(word_freq, key=word_freq.get, reverse=True)[:10]

#skipgram
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model= Word2Vec(min_count = 2,size = 300,workers= cores -1 , window = 3, sg = 1)

#create vocabulary
t = time()
w2v_model.build_vocab(words, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

#numbers of words in vocabulary 
w2v_model.wv.vectors.shape #17 034 sentences, 11 647 sent #16987 sentences 11 632 sent

w2c = dict()
for item in w2v_model.wv.vocab:
    w2c[item]=w2v_model.wv.vocab[item].count

#w2cSorted=dict(sorted(w2c.items(), key=lambda x: x[1],reverse=True))

t = time()
w2v_model.train(sent, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
 
w2v_model['septický']

cosine_distance (w2v_model,'biologie',w2c,3)

display_closestwords_tsnescatterplot(w2v_model, 'biologie', 300)
display_closestwords_tsnescatterplot(w2v_model, 'datum', 300)
display_closestwords_tsnescatterplot(w2v_model, 'oko', 300) #adnex, adnexa 
display_closestwords_tsnescatterplot(w2v_model, 'sval', 300) 
display_closestwords_tsnescatterplot(w2v_model, 'srdce', 300) 

w2v_model.similarity('srdce','sval')




# word2vec CBOW

cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model= Word2Vec(min_count = 2,size = 300,workers= cores -1 , window = 3, sg = 0)

#create vocabulary
t = time()
w2v_model.build_vocab(words, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

#numbers of words in vocabulary 
w2v_model.wv.vectors.shape #17 034 sentences, 11 647 sent #16987 sentences 11 632 sent

w2c = dict()
for item in w2v_model.wv.vocab:
    w2c[item]=w2v_model.wv.vocab[item].count

#w2cSorted=dict(sorted(w2c.items(), key=lambda x: x[1],reverse=True))
 
t = time()
w2v_model.train(words, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

 
w2v_model['septický']

cosine_distance (w2v_model,'biologie',w2c,3)

display_closestwords_tsnescatterplot(w2v_model, 'biologie', 300)
display_closestwords_tsnescatterplot(w2v_model, 'datum', 300)
display_closestwords_tsnescatterplot(w2v_model, 'oko', 300) #adnex, adnexa 
display_closestwords_tsnescatterplot(w2v_model, 'sval', 300) 
display_closestwords_tsnescatterplot(w2v_model, 'srdce', 300) 

w2v_model.similarity('srdce','sval')


### FASTTEXT skipgram
#import fasttext
from gensim.models.fasttext import FastText 
#ftxt= gensim.models.FastText(sent, min_count=2,size= 300,workers=3, window =3, sg = 1)
ftxt = FastText(size=300, window=3, min_count=2, workers = 3, sg = 1)  # instantiate
ftxt.build_vocab(sentences=words)
ftxt.train(sentences=words, total_examples=len(sent), epochs=10)  # train

#ftxt= fasttext.load_model('cc.cs.300.bin')
#t = time()

display_closestwords_tsnescatterplot(ftxt, 'šok', 300)
display_closestwords_tsnescatterplot(ftxt, 'datum', 300)
display_closestwords_tsnescatterplot(ftxt, 'oko', 300) #adnex, adnexa 
display_closestwords_tsnescatterplot(ftxt, 'sval', 300) 
display_closestwords_tsnescatterplot(ftxt, 'srdce', 300) 

ftxt.similarity('vagina','penis')


### CBOW
ftxt = FastText(size=300, window=3, min_count=2, workers = 3, sg = 0)  # instantiate
ftxt.build_vocab(sentences=words)
ftxt.train(sentences=words, total_examples=len(sent), epochs=10)  # train

display_closestwords_tsnescatterplot(ftxt, 'šok', 300)
display_closestwords_tsnescatterplot(ftxt, 'datum', 300)
display_closestwords_tsnescatterplot(ftxt, 'oko', 300) #adnex, adnexa 
display_closestwords_tsnescatterplot(ftxt, 'sval', 300) 
display_closestwords_tsnescatterplot(ftxt, 'srdce', 300) 

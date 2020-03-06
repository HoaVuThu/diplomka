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
from stop_words import get_stop_words
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

#data preprocessing
load_reporting= clean_data('.\\reporting\\reporting-without-comma.json', '.\\reporting\\Reporting_Xpath_vol2.xlsx')
reporting = []
for entity in load_reporting:
    reporting.append(entity['elements']) ### pomenit kod tu potom aby upravy prebiehali v json 

#kontrola ci sa vobec nieco vytiahlo     
list2 = [x for x in reporting if x]    
    
reporting2 = []
for line in reporting:
    reporting2.append(" ".join(line))
    
del reporting, load_reporting, entity, line

#reporting_preproc = word_preprocessing(reporting2)# 1 119 619 words
reporting_preproc = sentence_preprocessing(reporting2)# 1 908 documents

reporting_UDPipe = [UDPipe_preprocessing_word(ls) for ls in reporting_preproc] #NS -> n

for ls in reporting_UDPipe: # rucne zasiahnutie zle nalemmatizovanych slov 
    ls.replace('moci', 'nemoci')


del ls

##### word2vec
sent = [word.split() for word in reporting_UDPipe]
phrases = Phrases(sent, min_count=2, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]

word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
len(word_freq) #16 028

sorted(word_freq, key=word_freq.get, reverse=True)[:10]

cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model= gensim.models.Word2Vec(min_count=2,size= 300,workers=3, window =3, sg = 1) #nastrel

#create vocabulary
t = time()
w2v_model.build_vocab(sentences, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

#number of words in vocabulary
w2v_model.wv.vectors.shape #14019 sentences, 9390 sent 

w2c = dict()
for item in w2v_model.wv.vocab:
    w2c[item]=w2v_model.wv.vocab[item].count

w2cSorted=dict(sorted(w2c.items(), key=lambda x: x[1],reverse=True))


#training
t = time()
w2v_model.train(sent, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2))) # ~ 2min 4 sek 

 
w2v_model['septický']

cosine_distance (w2v_model,'biologie',w2c,3)

display_closestwords_tsnescatterplot(w2v_model, 'šok', 300)
display_closestwords_tsnescatterplot(w2v_model, 'datum', 300)
display_closestwords_tsnescatterplot(w2v_model, 'oko', 300) #adnex, adnexa 
display_closestwords_tsnescatterplot(w2v_model, 'sval', 300) 
display_closestwords_tsnescatterplot(w2v_model, 'srdce', 300) 

w2v_model.similarity('srdce','sval')


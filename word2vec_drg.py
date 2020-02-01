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
# from stop_words import get_stop_words
# from gensim.models import Word2Vec
# from gensim.models.phrases import Phrases, Phraser
import logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

from def_clean_data import clean_data

#data preprocessing
load_drg = clean_data('.\\drg\\drg-without-comma.json', '.\\drg\\DRG-klasifikace-XPath-vol9_nocodes.xlsx')
drg = []
for entity in load_drg:
    drg.append(entity['elements']) ### pomenit kod tu potom aby upravy prebiehali v json 
    
drg2 = []
for line in drg:
    drg2.append(" ".join(line))
    
del drg, load_drg, entity, line

#drg_preproc = word_preprocessing(drg2)# 1 119 619 words
drg_preproc = sentence_preprocessing(drg2)# 1 908 documents

drg_UDPipe = [UDPipe_preprocessing_word(ls) for ls in drg_preproc] #NS -> n

for ls in drg_UDPipe: # rucne zasiahnutie zle nalemmatizovanych slov 
    ls.replace('moci', 'nemoci')
'''
předloktit
nadloktit
stehný
n 
zn
neurč
čt_osý
odpočinek_jídlo
zra
urč
provnat_neh
nepr_neh
neh
on_měkký
NÚ
noho
novotvar_nnch
Ný 
provoz_neh
mot
tonutí_potop
infekč
pagetuv
vyst
ca_in
LAPAROSKOPICKY
pev_přka
želez
řid
neprovnat_neh
Nebo
kompl
výst
hermatros
bus_zrá
blaný
dodavý
lát
bakter
bus
RESEKCE
exp
čéška
parazit_on
oba
on
druhý
dopr
vytv
ucp_dý
vd_polk
cho_škoda
novotvar_NNCH
vyst_kontr
vystv_nekontr
přesně
ý
hk_DK
kot
AORTY
Jednostranný
CÉVOU_VZP
sek_neurč
říd_kol
Protetický_VZP
první
druhý
třetí
sponto_roztrž
do
v_STACIONÁŘI
Autologní_CÉVOU
BYPASS_FEMORO
edý
zbr
EXCIZE_Exstirpace
pr
'''
del ls

##### word2vec
sent = [word.split() for word in drg_UDPipe]
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

#size - the number of dimension of the embeddings, def = 100
#window - the maximum distance between a target word and words around the target word, def = 5
#min_count - the minimum count of words to consider when training the model; words with occurrence less than this count will be ignored, def = 5
#workers - the number of partitions during training , def = 3
#sg - the training algorithm, CBOW - 0, skipgram - 1, defaul = CBOW
 
w2v_model['septický']
''' 
def cosine_distance (model, word,target_list , num) :
    cosine_dict ={}
    word_list = []
    a = model[word]
    for item in target_list :
        if item != word :
            b = model [item]
            cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
            cosine_dict[item] = cos_sim
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]

lek_disc = list(vocabulary.keys()) 

cosine_distance (w2v_model,'biologie',w2c,3)
'''

def display_closestwords_tsnescatterplot(model, word, size):
    
    arr = np.empty((0,size), dtype='f')
    word_labels = [word]
    
    close_words = model.similar_by_word(word)
    
    arr = np.append(arr, np.array([model[word]]), axis=0)

    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
            
        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()

display_closestwords_tsnescatterplot(w2v_model, 'šok', 300)
display_closestwords_tsnescatterplot(w2v_model, 'datum', 300)
display_closestwords_tsnescatterplot(w2v_model, 'oko', 300) #adnex, adnexa 
display_closestwords_tsnescatterplot(w2v_model, 'sval', 300) 
display_closestwords_tsnescatterplot(w2v_model, 'srdce', 300) 

w2v_model.similarity('srdce','sval')


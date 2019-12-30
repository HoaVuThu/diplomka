import json
import codecs
import pandas as pd
import re
import numpy as np
from time import time
from collections import defaultdict
import multiprocessing
from stop_words import get_stop_words
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


'''load_drg = clean_data('.\\drg\\drg-without-comma.json', '.\\drg\\DRG-klasifikace-XPath-vol9_nocodes.xlsx')
drg = []
for entity in load_drg:
    drg.append(entity['elements'])

drg = [UDPipe_preprocessing(line) for ls in drg for line in ls]

for ls in drg:
    if len(ls) == 0:
        drg.remove(ls)
    for word in ls:
        if word == "." or word == ")"  or word == ";" or word == "," or word == "-" or word in stop:
            ls.remove(word)
        if "(" in word:
            word = word.replace("(", "")

#count number of words 
number_of_words = 0
for line in drg:
    number_of_words += len(line)


#stop words
stop = get_stop_words('czech')
for i in ['a', 's', 'při', 'k', 'v', 'o', 'z', 'i']:
    stop.append(i)

#remove stop words   
for ls in drg:
    for word in ls:
        if word in stop:
            ls.remove(word)
        
number_of_words2 = 0
for line in drg:
    number_of_words2 += len(line)
                
with open('drg_clean_UDPipe_nocodes.txt', 'w', encoding = 'utf8') as outfile:
    json.dump(drg, outfile, ensure_ascii=False) #almost no codes, opravit XPATHS 

with open('drg_clean_UDPipe.txt', 'r', encoding = 'utf8') as file:
    drg = json.load(file)
'''

#data preprocessing

load_drg = clean_data('.\\drg\\drg-without-comma.json', '.\\drg\\DRG-klasifikace-XPath-vol9_nocodes.xlsx')
drg = []
for entity in load_drg:
    drg.append(entity['elements'])
    
drg2 = []
for line in drg:
    drg2.append(" ".join(line))

drg_preproc = text_preprocessing(drg2)# 1 218 438

drg_UDPipe = [UDPipe_preprocessing(ls) for ls in drg_preproc]
        
number_of_words_drg = 0

for line in drg2:
    splt = line.split()
    number_of_words_drg += len(splt)


##### word2vec

sent = [word.split() for ls in drg for word in ls]
phrases = Phrases(sent, min_count=2, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]

word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
len(word_freq)

sorted(word_freq, key=word_freq.get, reverse=True)[:10]

w2v_model= gensim.models.Word2Vec(drg, min_count=2,size= 300,workers=3, window =3, sg = 1)

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

t = time()

w2v_model.build_vocab(sentences, progress_per=10000)

#size - the number of dimension of the embeddings, def = 100
#window - the maximum distance between a target word and words around the target word, def = 5
#min_count - the minimum count of words to consider when training the model; words with occurrence less than this count will be ignored, def = 5
#workers - the number of partitions during training , def = 3
#sg - the training algorithm, CBOW - 0, skipgram - 1, defaul = CBOW
 
model['septický']
 
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

cosine_distance (model,'Biologie',lek_disc,3)

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
    plt.pyplot.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
            plt.pyplot.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()

display_closestwords_tsnescatterplot(w2v_model, 'šok', 300)
display_closestwords_tsnescatterplot(w2v_model, 'datum', 300)
display_closestwords_tsnescatterplot(w2v_model, 'oko', 300) #adnex, adnexa 
display_closestwords_tsnescatterplot(w2v_model, 'sval', 300) 
display_closestwords_tsnescatterplot(w2v_model, 'srdce', 300) 

w2v_model.similarity('srdce','sval')


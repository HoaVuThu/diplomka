import json
import pandas as pd
import dfply as df
import numpy as np
import matplotlib.pyplot as plt
from time import time
from collections import defaultdict
from sklearn.manifold import TSNE
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


test = pd.read_excel('jednotkyZaklad.xlsx')
test = test >> df.drop(df.contains('Rozsah'), )

test.columns = ['nazev_jednotky', 'specialista', 'garant', 'sekce', 'lek_disc', 'kurz_kod','kurz_nazov', 'vyznam', 'klic_slova', 'vyzn_pojmy', 'vystup']
test = test.replace(np.nan, "")
test = test >> df.drop('specialista', 'garant', 'kurz_kod')

# concanating all information into new created column 
test['all'] = ""
for name_column in test.columns:
    if name_column == "all":
        pass
    else:
        test['all'] = test['all'].str.cat(test[name_column],  sep=', ')

del name_column

test = test >> df.drop('nazev_jednotky', 'sekce', 'lek_disc', 'kurz_nazov', 'vyznam', 'klic_slova', 'vyzn_pojmy', 'vystup') #drop all column but "all" 
test['all'] = test['all'].str.replace(", ", "", n = 1) #remove first comma 

optimed = [words for words in test['all']] #convert documents into list 
optimed_preproc = sentence_preprocessing(optimed) #preprocessing
optimed_UDPipe = [UDPipe_preprocessing_word(ls) for ls in optimed_preproc] #tokenization + lemmatization 


with open('OPTIMED_clean_preprocessed.txt', 'w', encoding = 'utf8') as outfile:
    json.dump(optimed_UDPipe, outfile, ensure_ascii=False)         

with open('OPTIMED_clean_preprocessed.txt', 'r', encoding = 'utf8') as file:
    optimed_UDPipe = json.load(file)


sent = [word.split() for word in optimed_UDPipe]
phrases = Phrases(sent, min_count=2, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]

word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
len(word_freq) #27 852
 
sorted(word_freq, key=word_freq.get, reverse=True)[:20]

cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model= Word2Vec(min_count = 2,size = 300,workers= cores -1 , window = 3, sg = 1)

#create vocabulary
t = time()
w2v_model.build_vocab(sentences, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

#numbers of words in vocabulary 
w2v_model.wv.vectors.shape #17 034 sentences, 11 647 sent

w2c = dict()
for item in w2v_model.wv.vocab:
    w2c[item]=w2v_model.wv.vocab[item].count

w2cSorted=dict(sorted(w2c.items(), key=lambda x: x[1],reverse=True))

'''
tento
jednotlivý
včetně
abnormalita_tento
možný?
další
syntromě
být
zejména
mikrobiologický_mikrobiologický
vyšetření_podle
dle
podle
popsat_compartment
a
všechen
také
některý
jícno
vv
prsa
hormon
jeho
rr
často
et
jenž
který
be
on
mm`milimetr
us
ct
uvedený
ten
oblast_ci
také_mikroba
vs
svat
aa
moci
VJ
jeden
se
Mister
sekrec
apod
na
daný
být_mít
gll
jmenovat
nn
v
hodně
nový
velmi
CD
LD_km
ca_CRP
ct_CNS
Il_Il
edý
tj
CD_CD
druhý
jaký
lze
rovněž
Il
ie
rh
však
zvýšený
an
iga
in
jednak
nasi
ph
při
VVV
HCAI
atd
igg
pet
ae
chromosom
chromozom
již
mnoho
mimo
ca
de
karcien
mezi
mít_být
Gist
dg
ei
gl
hrtat
jet
jít
pomoci
stále
th
určený
varlat
ad
UZ
and_^(obv._souč._anglických_názvů,_"a"
ani
bez
buď
což
dít
fibrát
ig
igg_ig
kom
kromě
mít
obvykle
All_All
HLA_HLA
Malta_lymfom
P
P_ovzduší
dva
močův
ncl
ncíl
noh
nv
oogenézt
opláziit
opláziout
papit
sup
síný
tři
uhý_smysl
zcela
zda
zde
cca
gnrh
hcl
lymeský
male
naopak
no
néva
oba
pat
per
sak
sarkdo
též
up
XI
XII
ab
dalý_péče
devicův
digitý
gis
gingivo
igd
ire
jako
lební
mon
opět
p
pr
sem
semien
ser
seznámý
shnutový
Ch
Cth
Pq
Pnet
Pth
RAAS
Tia
_
mamm
*3
'''    

t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

#size - the number of dimension of the embeddings, def = 100
#window - the maximum distance between a target word and words around the target word, def = 5
#min_count - the minimum count of words to consider when training the model; words with occurrence less than this count will be ignored, def = 5
#workers - the number of partitions during training , def = 3
#sg - the training algorithm, CBOW - 0, skipgram - 1, defaul = CBOW
 
w2v_model['septický']
 
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


cosine_distance (w2v_model,'biologie',w2c,3)

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

display_closestwords_tsnescatterplot(w2v_model, 'biologie', 300)
display_closestwords_tsnescatterplot(w2v_model, 'datum', 300)
display_closestwords_tsnescatterplot(w2v_model, 'oko', 300) #adnex, adnexa 
display_closestwords_tsnescatterplot(w2v_model, 'sval', 300) 
display_closestwords_tsnescatterplot(w2v_model, 'srdce', 300) 

w2v_model.similarity('srdce','sval')

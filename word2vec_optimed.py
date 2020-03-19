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
from gensim.test.utils import get_tmpfile

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


#skipgram
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model= Word2Vec(min_count = 2,size = 300,workers= cores -1 , window = 3, sg = 1)

#create vocabulary
t = time()
w2v_model.build_vocab(sent, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

#numbers of words in vocabulary 
w2v_model.wv.vectors.shape #17 034 sentences, 11 647 sent #16987 sentences 11 632 sent

w2c = dict()
for item in w2v_model.wv.vocab:
    w2c[item]=w2v_model.wv.vocab[item].count

#w2cSorted=dict(sorted(w2c.items(), key=lambda x: x[1],reverse=True))

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


'''
pomocí
empyý = empyém
uvey  ok
traumat = traumata? 
ana = anus
duodený = duodenum
hrtat = hrtan
1189 ci = CT
    ct = CT
1186 Gita = GIT
propedeutika urologi  = Propedeutika v neurologii
nstudent = student 3 ######## vymazat /r/n
edý = edém
kom = koma
urologický komplikace = nuerologické komplikace
uromuskulární = neuromuskulární 
čití
syndromě = syndrom
erytematod = erytematodes
oplazma = neoplazma
pankreat = pankreas
varlést = varlata
kombinout = kombinovat
gll = Gll.
S?? 3 

'''
t = time()
w2v_model.train(sent, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

w2v_model.init_sims(replace=True)
w2v_model.save("word2vec_optimed.model")
w2v_model.save("word2vec_optimed.kv")


#size - the number of dimension of the embeddings, def = 100
#window - the maximum distance between a target word and words around the target word, def = 5
#min_count - the minimum count of words to consider when training the model; words with occurrence less than this count will be ignored, def = 5
#workers - the number of partitions during training , def = 3
#sg - the training algorithm, CBOW - 0, skipgram - 1, defaul = CBOW
 
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
w2v_model.build_vocab(sent, progress_per=10000)
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


### FASTTEXT skipgram
#import fasttext
from gensim.models.fasttext import FastText 
#ftxt= gensim.models.FastText(sent, min_count=2,size= 300,workers=3, window =3, sg = 1)
ftxt = FastText(size=300, window=3, min_count=2, workers = 3, sg = 1)  # instantiate
ftxt.build_vocab(sentences=sent)
ftxt.train(sentences=sent, total_examples=len(sent), epochs=10)  # train

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
ftxt.build_vocab(sentences=sent)
ftxt.train(sentences=sent, total_examples=len(sent), epochs=10)  # train

display_closestwords_tsnescatterplot(ftxt, 'šok', 300)
display_closestwords_tsnescatterplot(ftxt, 'datum', 300)
display_closestwords_tsnescatterplot(ftxt, 'oko', 300) #adnex, adnexa 
display_closestwords_tsnescatterplot(ftxt, 'sval', 300) 
display_closestwords_tsnescatterplot(ftxt, 'srdce', 300) 

ftxt.similarity('vagina','penis')

'S' in w2c
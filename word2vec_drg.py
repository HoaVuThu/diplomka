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
# from stop_words import get_stop_words
# from gensim.models import Word2Vec
# from gensim.models.phrases import Phrases, Phraser
import logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

#data preprocessing
load_drg = clean_data('.\\drg\\drg-without-comma.json', '.\\drg\\DRG-klasifikace-XPath-vol9_nocodes.xlsx')
drg = []
for entity in load_drg:
    drg.append(entity['elements']) ### pomenit kod tu potom aby upravy prebiehali v json 
    
for i, line in enumerate(drg):
    drg[i] = " ".join(line)
    
del i ,load_drg, entity, line

#drg_preproc = word_preprocessing(drg2)# 1 119 619 words
drg_preproc = sentence_preprocessing(drg)# 1 908 documents

drg_UDPipe = [UDPipe_preprocessing_word(ls) for ls in drg_preproc]


#with open('drg_clean_for_correction.txt', 'w', encoding = 'utf8') as outfile:
#    json.dump(drg_UDPipe, outfile, ensure_ascii=False)         

with open('drg_clean_for_correction.txt', 'r', encoding = 'utf8') as file:
    drg_UDPipe = json.load(file)

for i, word in enumerate(drg_UDPipe):# rucne zasiahnutie zle nalemmatizovanych slov .... treba regex na rozne tvary 
        drg_UDPipe[i] = word.lower()
 
#drg_UDPipe = [list(ls.split(" ")) for ls in drg_UDPipe]
        #700
#drg_UDPipe2 = drg_UDPipe
del i, word

for i in range(len(drg_UDPipe)):# rucne zasiahnutie zle nalemmatizovanych slov .... treba regex na rozne tvary 
        
        drg_UDPipe[i] = re.sub(' absces\w', ' absces', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' abdomináln\w* ', ' abdominální ', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' ablací', ' ablace', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' abnormal ', ' abnormální ', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' abnormalit', ' abnormalita', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' acetabula\w', ' acetabulum', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' adhe[s|z]í', ' adheze', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' adnex\w', ' adnex', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' aktinomykosis', ' aktinomykóza', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' alopecie', ' alopecium', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' alveolární\w*', ' alveolární', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' alveol[u]', ' alveola', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' ambulanca ', ' ambulance ', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' ambulantním', ' ambulantní', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' amniová', ' amniový', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' anaerob\w', ' anaeroba', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' anastom\w*', ' anastomóza', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' anest[e|é]\w*', ' anestézie', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' aneury[s|z]\w*', ' aneuryzma', drg_UDPipe[i], flags=re.UNICODE) #aneuryzmatický
        drg_UDPipe[i] = re.sub(' angiostrongyliasis', ' angiostrongylióza', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' ankylóz[y|a]', ' ankylóza', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' anomálního', ' anomální', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' anorektálního', ' anorektální', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' antipyretika', ' antipyretikum', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' análního', ' anální', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' aortofemorálním', ' aortofemorální', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' aort[u|y|ě]', ' aorta', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' apendicitidě', ' apendicitida', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' apl[a|á]azi\w', ' aplazie', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' aplikací', ' aplikace', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' arteriosus', ' artetriální', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' arteri[a|e]\w*', ' arterie', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' arthritis', ' artritida', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' arthrodéza', ' artrodéza', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' arytmického', ' arytmický', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' atr[é|e][s|z]i[i|e]', ' atrézie', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' atrioventrikulárního', ' atrioventrikulární', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' atypická', ' atypický', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' autotransplantací', ' autotransplantace', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' autotransplantát\w*', ' autotransplantát', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' arthrodéza', ' artrodéza', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' arthrodéza', ' artrodéza', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' autologní\w*', ' autologní', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' axil[o|y]', ' axily', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' azbesz', ' azbest', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' moci ', ' nemoc ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' předloktit ', ' předloktí ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' nadloktit ', ' nadloktí ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' pažet ', ' paže ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' neurč ', ' neurčený ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' stehný ', ' stehno ', drg_UDPipe[i], flags=re.UNICODE)     ###
        drg_UDPipe[i] = re.sub(' n ', ' ns ', drg_UDPipe[i], flags=re.UNICODE)    
#        drg_UDPipe[i] = re.sub(' zn ', ' ZN ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' on ', ' onemocnění ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' igg ', '', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' iga ', '', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' igm ', '', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' ig ', '', drg_UDPipe[i], flags=re.UNICODE)    
#        drg_UDPipe[i] = re.sub(' nnch', 'NNCH', drg_UDPipe[i])
#        drg_UDPipe[i] = re.sub(' Odtd', 'ODTD', drg_UDPipe[i])
        drg_UDPipe[i] = re.sub(' mk ', '', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' mg ', '', drg_UDPipe[i], flags=re.UNICODE)    
#        drg_UDPipe[i] = re.sub(' KIU ', '', drg_UDPipe[i], flags=re.UNICODE)    
#        drg_UDPipe[i] = re.sub(' IU ', '', drg_UDPipe[i], flags=re.UNICODE)    
#        drg_UDPipe[i] = re.sub(' af ', '', drg_UDPipe[i], flags=re.UNICODE)    
 #       drg_UDPipe[i] = re.sub(' dik ', ' DIK ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' mladý ', '', drg_UDPipe[i], flags=re.UNICODE)    
 #       drg_UDPipe[i] = drg_UDPipe[i].replace('por', 'porucha')
 #       drg_UDPipe[i] = drg_UDPipe[i].replace('duš', 'duševní')
 #       drg_UDPipe[i] = re.sub(" gamunex", 'gamunex', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(" laparoskopi(.)", 'laparoskopie', drg_UDPipe[i], flags=re.UNICODE)    
  #      drg_UDPipe[i] = re.sub(' hk ', ' HK ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' kr ', '', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' antiastmatika ', ' antiasmatikum ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' karboanhydráz ', ' karboanhydrázy ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' benzenuenzený ', ' benzenuenzen ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' benzenuý ', ' benzen ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' chloroforma ', ' chloroform ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub('etylena ', 'etylen ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' olův ', ' olovo ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' arsenuý ', ' arsen ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' edý ', ' edém ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' ab ', '', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' rh ', '', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' nepříza ', ' nepříznivý ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' bázet ', ' báze ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' zostro ', ' zoster ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' zn[ý]* ', ' zn ', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' sklér ', ' skléra ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' modalit ', ' modality ', drg_UDPipe[i], flags=re.UNICODE)    
        drg_UDPipe[i] = re.sub(' biops\w*', ' biopsie', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' biopie', ' biopsie', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' bypass\w*', ' bypass', drg_UDPipe[i], flags=re.UNICODE)
        drg_UDPipe[i] = re.sub(' chirurgick\w*', ' chirurgicky', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' chlopn\w*', ' chlopně', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' chrupavk\w*', ' chrupavka', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' CR ', ' čr ', drg_UDPipe[i], flags=re.UNICODE)     
        drg_UDPipe[i] = re.sub(' cyst[u|y]', ' cysta', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' cévou\w*', ' céva', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' cévní\w*', ' cévní', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' defibrilačn\w*', ' defibrilační', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' defibrilátor\w*', ' defibrilátor', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' dekompres\w*', ' dekomprese', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' diastáz\w*', ' diastáze', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' dilatac\w*', ' dilatace', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' disci[s|z]e', ' discize', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' disekc\w', ' disekce', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' drenáž\w', ' drenáž', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' endoprotéz\w', ' endoprotéza', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' exci[s|z]\w', ' excize', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' exostó\w', ' exostóza', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' fasc[i,e]*', ' fascie', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' femoral[e|i]s', ' femoralis', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' femor[o|u]', ' femoro', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' femur\w*', ' femur', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' flexor\w*', ' flexor', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' fragment\w*', ' fragment', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' glaukom\w*', ' glaukom', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' hobbard\w*', ' hobbardov', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' inci[s|z]\w', ' incize', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' jícnov\w*', ' jícnový', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' jícnu\w*', ' jícen', drg_UDPipe[i], flags=re.UNICODE)       
        drg_UDPipe[i] = re.sub(' inci[s|z]\w', ' incize', drg_UDPipe[i], flags=re.UNICODE)       


data = [list(ls.split(" ")) for ls in drg_UDPipe]

with open('drg_clean_preprocessed.txt', 'w', encoding = 'utf8') as outfile:
    json.dump(data, outfile, ensure_ascii=False)         

with open('drg_clean_preprocessed.txt', 'r', encoding = 'utf8') as file:
    drg_UDPipe = json.load(file)

'''
JEJUNA
JEJUNEM
JEJUNO
Immunást
IU
IVUS
IO
IOČ
Horizontálň
GYNEKOLOGICKOU
GENOVOU
GJ`poul
FVIII
femoro
Endovasálnit
Eventuelno
Dermato(poly)myozitida - dermatý pol myozitida
Záknout - ZÁKL. 
zra
neh
NÚ
noho
Ný 
mot
vyst
řid
hermatros
blaný
dodavý
bus
exp
čéška
dopr
vytv
ý
kot
do
edý
zbr
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
w2v_model.build_vocab(sent, progress_per=10000)
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

display_closestwords_tsnescatterplot(w2v_model, 'šok', 300)
display_closestwords_tsnescatterplot(w2v_model, 'datum', 300)
display_closestwords_tsnescatterplot(w2v_model, 'oko', 300) #adnex, adnexa 
display_closestwords_tsnescatterplot(w2v_model, 'sval', 300) 
display_closestwords_tsnescatterplot(w2v_model, 'srdce', 300) 

w2v_model.similarity('srdce','sval')

### FASTTEXT skipgram
ftxt= gensim.models.FastText(drg, min_count=2,size= 300,workers=3, window =3, sg = 1)
ftxt= fasttext.load_model('cc.cs.300.bin')

t = time()
ftxt.build_vocab(sentences, progress_per=10000)

display_closestwords_tsnescatterplot(ftxt, 'šok', 300)
display_closestwords_tsnescatterplot(ftxt, 'datum', 300)
display_closestwords_tsnescatterplot(ftxt, 'oko', 300) #adnex, adnexa 
display_closestwords_tsnescatterplot(ftxt, 'sval', 300) 
display_closestwords_tsnescatterplot(ftxt, 'srdce', 300) 

ftxt.similarity('vagina','penis')


### CBOW
ftxt= gensim.models.FastText(drg, min_count=2,size= 300,workers=3, window =3, sg = 0)
ftxt= fasttext.load_model('cc.cs.300.bin')

t = time()
ftxt.build_vocab(sentences, progress_per=10000)

display_closestwords_tsnescatterplot(ftxt, 'šok', 300)
display_closestwords_tsnescatterplot(ftxt, 'datum', 300)
display_closestwords_tsnescatterplot(ftxt, 'oko', 300) #adnex, adnexa 
display_closestwords_tsnescatterplot(ftxt, 'sval', 300) 
display_closestwords_tsnescatterplot(ftxt, 'srdce', 300) 

ftxt.similarity('vagina','penis')
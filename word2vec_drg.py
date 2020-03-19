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
# from gensim.models import Word2Vec
# from gensim.models.phrases import Phrases, Phraser
import logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

#data preprocessing
drg = clean_data('.\\drg\\drg-without-comma.json', '.\\drg\\DRG-klasifikace-XPath-vol9_nocodes.xlsx')
#drg = []
#for entity in load_drg:
#    drg.append(entity['elements']) ### pomenit kod tu potom aby upravy prebiehali v json 
#    
#for i, line in enumerate(drg):
#    drg[i] = " ".join(line)
#    
#del i ,load_drg, entity, line

#drg_preproc = word_preprocessing(drg2)# 1 119 619 words

for entity in drg:
    entity['elements'] = [UDPipe_preprocessing_word(ls) for ls in entity['elements']]

for entity in drg:
    entity['elements'] = sentence_preprocessing(entity['elements'])# 1 908 documents

#drg_UDPipe = [UDPipe_preprocessing_word(ls) for ls in drg_preproc]

with open('drg_clean_for_correction_json.txt', 'w', encoding = 'utf8') as outfile:
    json.dump(drg, outfile, ensure_ascii=False)         

with open('drg_clean_for_correction_json.txt', 'r', encoding = 'utf8') as file:
    drg_UDPipe = json.load(file)

#for i, word in enumerate(drg_UDPipe):
#        drg_UDPipe[i] = word.lower()

#del i, word

for entity in drg:
    for i in range(len(entity['elements'])):# rucne zasiahnutie zle nalemmatizovanych slov .... treba regex na rozne tvary     
        entity['elements'][i] = re.sub(' abscesw', ' absces', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' abdominálnw* ', ' abdominální ', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' ablací', ' ablace', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' abnormal ', ' abnormální ', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' abnormalit', ' abnormalita', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' acetabulaw', ' acetabulum', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' adhe[s|z][i]í', ' adheze', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' adnexw', ' adnex', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' aktinomykosis', ' aktinomykóza', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' alopecie', ' alopecium', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' alveolárníw*', ' alveolární', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' alveol[u][i]', ' alveola', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' ambulanca ', ' ambulance ', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' ambulantním', ' ambulantní', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' amniová', ' amniový', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' anaerobw', ' anaeroba', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' anastomw*', ' anastomóza', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' anest[e|é][i]w*', ' anestézie', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' aneury[s|z][i]w*', ' aneuryzma', entity['elements'][i], flags=re.UNICODE) #aneuryzmatický
        entity['elements'][i] = re.sub(' angiostrongyliasis', ' angiostrongylióza', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' ankylóz[y|a][i]', ' ankylóza', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' anomálního', ' anomální', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' anorektálního', ' anorektální', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' antipyretika', ' antipyretikum', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' análního', ' anální', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' aortofemorálním', ' aortofemorální', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' aort[u|y|ě][i]', ' aorta', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' apendicitidě', ' apendicitida', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' apl[a|á][i]aziw', ' aplazie', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' aplikací', ' aplikace', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' arteriosus', ' artetriální', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' arteri[a|e][i]w*', ' arterie', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' arthritis', ' artritida', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' arthrodéza', ' artrodéza', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' arytmického', ' arytmický', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' atr[é|e][i][s|z][i]i[i|e][i]', ' atrézie', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' atrioventrikulárního', ' atrioventrikulární', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' atypická', ' atypický', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' autotransplantací', ' autotransplantace', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' autotransplantátw*', ' autotransplantát', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' arthrodéza', ' artrodéza', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' arthrodéza', ' artrodéza', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' autologníw*', ' autologní', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' axil[o|y][i]', ' axily', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' azbesz', ' azbest', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' moci ', ' nemoc ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' předloktit ', ' předloktí ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' nadloktit ', ' nadloktí ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' pažet ', ' paže ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' neurč ', ' neurčený ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' stehný ', ' stehno ', entity['elements'][i], flags=re.UNICODE)     ###
        entity['elements'][i] = re.sub(' n ', ' ns ', entity['elements'][i], flags=re.UNICODE)    
#        entity['elements'][i] = re.sub(' zn ', ' ZN ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' on ', ' onemocnění ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' igg ', '', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' iga ', '', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' igm ', '', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' ig ', '', entity['elements'][i], flags=re.UNICODE)    
#        entity['elements'][i] = re.sub(' nnch', 'NNCH', entity['elements'][i])
#        entity['elements'][i] = re.sub(' Odtd', 'ODTD', entity['elements'][i])
        entity['elements'][i] = re.sub(' mk ', '', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' mg ', '', entity['elements'][i], flags=re.UNICODE)    
#        entity['elements'][i] = re.sub(' KIU ', '', entity['elements'][i], flags=re.UNICODE)    
#        entity['elements'][i] = re.sub(' IU ', '', entity['elements'][i], flags=re.UNICODE)    
#        entity['elements'][i] = re.sub(' af ', '', entity['elements'][i], flags=re.UNICODE)    
 #       entity['elements'][i] = re.sub(' dik ', ' DIK ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' mladý ', '', entity['elements'][i], flags=re.UNICODE)    
 #       entity['elements'][i] = entity['elements'][i].replace('por', 'porucha')
 #       entity['elements'][i] = entity['elements'][i].replace('duš', 'duševní')
 #       entity['elements'][i] = re.sub(" gamunex", 'gamunex', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(" laparoskopi(.)", 'laparoskopie', entity['elements'][i], flags=re.UNICODE)    
  #      entity['elements'][i] = re.sub(' hk ', ' HK ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' kr ', '', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' antiastmatika ', ' antiasmatikum ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' karboanhydráz ', ' karboanhydrázy ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' benzenuenzený ', ' benzenuenzen ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' benzenuý ', ' benzen ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' chloroforma ', ' chloroform ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub('etylena ', 'etylen ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' olův ', ' olovo ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' arsenuý ', ' arsen ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' edý ', ' edém ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' ab ', '', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' rh ', '', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' nepříza ', ' nepříznivý ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' bázet ', ' báze ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' zostro ', ' zoster ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' zn[ý][i]* ', ' zn ', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' sklér ', ' skléra ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' modalit ', ' modality ', entity['elements'][i], flags=re.UNICODE)    
        entity['elements'][i] = re.sub(' biopsw*', ' biopsie', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' biopie', ' biopsie', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' bypassw*', ' bypass', entity['elements'][i], flags=re.UNICODE)
        entity['elements'][i] = re.sub(' chirurgickw*', ' chirurgicky', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' chlopnw*', ' chlopně', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' chrupavkw*', ' chrupavka', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' CR ', ' čr ', entity['elements'][i], flags=re.UNICODE)     
        entity['elements'][i] = re.sub(' cyst[u|y][i]', ' cysta', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' cévouw*', ' céva', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' cévníw*', ' cévní', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' defibrilačnw*', ' defibrilační', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' defibrilátorw*', ' defibrilátor', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' dekompresw*', ' dekomprese', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' diastázw*', ' diastáze', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' dilatacw*', ' dilatace', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' disci[s|z][i]e', ' discize', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' disekcw', ' disekce', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' drenážw', ' drenáž', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' endoprotézw', ' endoprotéza', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' exci[s|z][i]w', ' excize', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' exostów', ' exostóza', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' fasc[i,e][i]*', ' fascie', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' femoral[e|i][i]s', ' femoralis', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' femor[o|u][i]', ' femoro', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' femurw*', ' femur', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' flexorw*', ' flexor', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' fragmentw*', ' fragment', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' glaukomw*', ' glaukom', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' hobbardw*', ' hobbardov', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' inci[s|z][i]w', ' incize', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' jícnovw*', ' jícnový', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' jícnuw*', ' jícen', entity['elements'][i], flags=re.UNICODE)       
        entity['elements'][i] = re.sub(' inci[s|z][i]w', ' incize', entity['elements'][i], flags=re.UNICODE)       

for entity in drg:
    entity['elements'] = " ".join(entity['elements'])

for entity in drg:
    entity['elements'] = list(entity['elements'].split())
        
    
with open('drg_clean_preprocessed_json.txt', 'w', encoding = 'utf8') as outfile:
    json.dump(drg, outfile, ensure_ascii=False)         

with open('drg_clean_preprocessed_json.txt', 'r', encoding = 'utf8') as file:
    drg_UDPipe = json.load(file)

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
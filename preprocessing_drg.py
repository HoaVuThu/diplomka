import json
import re
import numpy as np
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

from def_udpipe import UDPipe_preprocessing_word
from def_sentence_preprocessing import sentence_preprocessing
from def_clean_data import clean_data
from def_word2vec_viz import tsne_plot_similar_words
import logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

#data preprocessing
drg = clean_data('.\\drg\\drg-without-comma.json', '.\\drg\\DRG-klasifikace-XPath-vol9_nocodes.xlsx')

for entity in drg:
    entity['elements'] = [UDPipe_preprocessing_word(ls) for ls in entity['elements']]

for entity in drg:
    entity['elements'] = sentence_preprocessing(entity['elements'])# 1 908 documents
    

#tuna - T 
with open('drg_clean_for_correction_json_reverseudpipe and sentence.txt', 'w', encoding = 'utf8') as outfile:
    json.dump(drg, outfile, ensure_ascii=False)         

with open('drg_clean_for_correction_json_reverseudpipe and sentence.txt', 'r', encoding = 'utf8') as file:
    drg = json.load(file)

for entity in drg:
    for i in range(len(entity['elements'])):# rucne zasiahnutie zle nalemmatizovanych slov .... treba regex na rozne tvary     
        entity['elements'][i] = re.sub(' moci ', ' nemoc ', entity['elements'][i], flags = re.UNICODE)    
        entity['elements'][i] = re.sub(' předloktit ', ' předloktí ', entity['elements'][i], flags = re.UNICODE)    
        entity['elements'][i] = re.sub(' nadloktit ', ' nadloktí ', entity['elements'][i], flags = re.UNICODE)
        entity['elements'][i] = re.sub(' zápěsit ', ' zápěstí ', entity['elements'][i], flags = re.UNICODE)
        entity['elements'][i] = re.sub(' pažet ', ' paže ', entity['elements'][i], flags = re.UNICODE)    
        entity['elements'][i] = re.sub(' stehný ', ' stehno ', entity['elements'][i], flags = re.UNICODE)     ###
        entity['elements'][i] = re.sub(' osteopora óza ', ' osteoporóza ', entity['elements'][i], flags = re.UNICODE)     ###
        entity['elements'][i] = re.sub(' tuna ', '', entity['elements'][i], flags = re.UNICODE)     ###
        entity['elements'][i] = re.sub(' kapský ', '', entity['elements'][i], flags = re.UNICODE)     ###
        entity['elements'][i] = re.sub(' celsius ', '', entity['elements'][i], flags = re.UNICODE)     ###
        entity['elements'][i] = re.sub(' metr ', '', entity['elements'][i], flags = re.UNICODE)
        entity['elements'][i] = re.sub(' rok ', '', entity['elements'][i], flags = re.UNICODE)
       

for entity in drg:
    entity['elements'] = " ".join(entity['elements'])

for entity in drg:
    entity['elements'] = list(entity['elements'].split())
        
    
with open('drg_clean_preprocessed_json_UDPipe_manual_correction.txt', 'w', encoding = 'utf8') as outfile:
    json.dump(drg, outfile, ensure_ascii=False)         

with open('drg_clean_preprocessed_json_UDPipe_manual_correction.txt', 'r', encoding = 'utf8') as file:
    drg_UDPipe = json.load(file)

##### word2vec
min_cnts = [2, 5]
architectures = [0, 1]
activations = [0, 1]
negatives = [5, 10, 15, 20]
cbow_mn = [0, 1]


for min_cnt in min_cnts:
    for architecture in architectures:
        if architecture  == 1:
            for activation in activations:
                if activation == 0:
                    for negat in negatives:
                        
                        words = []
                        
                        for entity in drg_UDPipe:
                            words.append(entity['elements'])
                                
                        w2v_model= Word2Vec(min_count = min_cnt, size = 300, window = 3,  workers = 3, sg = architecture, hs = activation, negative = negat, seed = 42)  
                        w2v_model.build_vocab(sentences = words)
                        w2v_model.train(sentences = words, total_examples = len(words), epochs = 30) 
                    
                        keys = ['sval', 'oko', 'srdce']
                    
                        embedding_clusters = []
                        word_clusters = []
                        for word in keys:
                            embeddings = []
                            words = []
                            for similar_word, _ in w2v_model.most_similar(word, topn=30):
                                words.append(similar_word)
                                embeddings.append(w2v_model[similar_word])
                            embedding_clusters.append(embeddings)
                            word_clusters.append(words)
                        
                        embedding_clusters = np.array(embedding_clusters)
                        n, m, k = embedding_clusters.shape
                        tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
                        embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                        
                        tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                                f'Sval oko srdce min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) +', hs = ' + str(activation) + ', negative = '+ str(negat) + ',  seed = 42.png')    
                else:
                    words = []
                        
                    for entity in drg_UDPipe:
                        words.append(entity['elements'])
                        
                    w2v_model= Word2Vec(min_count = min_cnt, size = 300, window = 3,  workers = 3, sg = architecture, hs = activation, seed = 42)  
                    w2v_model.build_vocab(sentences = words)
                    w2v_model.train(sentences = words, total_examples = len(words), epochs = 30) 
                
                    keys = ['sval', 'oko', 'srdce']
                
                    embedding_clusters = []
                    word_clusters = []
                    
                    for word in keys:
                        embeddings = []
                        words = []
                        
                        for similar_word, _ in w2v_model.most_similar(word, topn = 30):
                            words.append(similar_word)
                            embeddings.append(w2v_model[similar_word])
                        
                        embedding_clusters.append(embeddings)
                        word_clusters.append(words)
                    
                    embedding_clusters = np.array(embedding_clusters)
                    n, m, k = embedding_clusters.shape
                    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
                    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                    
                    tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                            f'Sval oko srdce min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) +', hs = ' + str(activation) + ',  seed = 42.png')    
                        
        else:
            for cbow in cbow_mn:
                for activation in activations:
                    if activation == 0:
                        for negat in negatives:
                                                    
                            words = []
                        
                            for entity in drg_UDPipe:
                                words.append(entity['elements'])
                                    
                            w2v_model = Word2Vec(min_count = min_cnt, size = 300, window = 3,  workers = 3, sg = architecture, cbow_mean = cbow, hs = activation, negative = negat, seed = 42)  
                            w2v_model.build_vocab(sentences = words)
                            w2v_model.train(sentences = words, total_examples = len(words), epochs = 30) 
                        
                            keys = ['sval', 'oko', 'srdce']
                        
                            embedding_clusters = []
                            word_clusters = []
                            
                            for word in keys:
                                embeddings = []
                                words = []
                                for similar_word, _ in w2v_model.most_similar(word, topn=30):
                                    words.append(similar_word)
                                    embeddings.append(w2v_model[similar_word])
                                embedding_clusters.append(embeddings)
                                word_clusters.append(words)
                            
                            embedding_clusters = np.array(embedding_clusters)
                            n, m, k = embedding_clusters.shape
                            tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
                            embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                            
                            tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                                    f'Sval oko srdce min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) + ', cbow_mean = ' + str(cbow) +', hs = ' + str(activation) + ', negative = '+ str(negat) + ',  seed = 42.png')    
                    else:
                        
                        words = []
                        
                        for entity in drg_UDPipe:
                            words.append(entity['elements'])
                            
                        w2v_model= Word2Vec(min_count = min_cnt, size = 300, window = 3,  workers = 3, sg = architecture, cbow_mean = cbow, hs = activation, seed = 42)  
                        w2v_model.build_vocab(sentences=words)
                        w2v_model.train(sentences=words, total_examples=len(words), epochs=30) 
                    
                        keys = ['sval', 'oko', 'srdce']
                    
                        embedding_clusters = []
                        word_clusters = []
                        
                        for word in keys:
                            embeddings = []
                            words = []
                            
                            for similar_word, _ in w2v_model.most_similar(word, topn=30):
                                words.append(similar_word)
                                embeddings.append(w2v_model[similar_word])
                           
                            embedding_clusters.append(embeddings)
                            word_clusters.append(words)
                        
                        embedding_clusters = np.array(embedding_clusters)
                        n, m, k = embedding_clusters.shape
                        tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
                        embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                        
                        tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                                f'Sval oko srdce min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) + ', cbow_mean = ' + str(cbow) +', hs = ' + str(activation) + ',  seed = 42.png')    
                            

### FASTTEXT skipgram
#import fasttext
from gensim.models.fasttext import FastText 
#ftxt= gensim.models.FastText(sent, min_count=2,size= 300,workers=3, window =3, sg = 1) 


min_cnts = [2, 5]
architectures = [0, 1]
activations = [0, 1]
negatives = [5, 10, 15, 20]
cbow_mn = [0, 1]
training_loss = []


for min_cnt in min_cnts:
    for architecture in architectures:
        if architecture  == 1:
            for activation in activations:
                if activation == 0:
                    for negat in negatives:
                        
                        words = []
                        
                        for entity in drg_UDPipe:
                            words.append(entity['elements'])
                                
                        ftxt= FastText(min_count = min_cnt, size = 300, window = 3,  workers = 3, sg = architecture, hs = activation, negative = negat, seed = 42)  
                        ftxt.build_vocab(sentences = words)
                        ftxt.train(sentences = words, total_examples = len(words), epochs = 30) 
                    
                        keys = ['sval', 'oko', 'srdce']
                    
                        embedding_clusters = []
                        word_clusters = []
                        for word in keys:
                            embeddings = []
                            words = []
                            for similar_word, _ in ftxt.most_similar(word, topn=30):
                                words.append(similar_word)
                                embeddings.append(ftxt[similar_word])
                            embedding_clusters.append(embeddings)
                            word_clusters.append(words)
                        
                        embedding_clusters = np.array(embedding_clusters)
                        n, m, k = embedding_clusters.shape
                        tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
                        embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                        
                        tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                                f'Sval oko srdce FastText min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) +', hs = ' + str(activation) + ', negative = '+ str(negat) + ',  seed = 42.png')    
                else:
                    words = []
                        
                    for entity in drg_UDPipe:
                        words.append(entity['elements'])
                        
                    ftxt= FastText(min_count = min_cnt, size = 300, window = 3,  workers = 3, sg = architecture, hs = activation, seed = 42)  
                    ftxt.build_vocab(sentences = words)
                    ftxt.train(sentences = words, total_examples = len(words), epochs = 30) 
                
                    keys = ['sval', 'oko', 'srdce']
                
                    embedding_clusters = []
                    word_clusters = []
                    
                    for word in keys:
                        embeddings = []
                        words = []
                        
                        for similar_word, _ in ftxt.most_similar(word, topn = 30):
                            words.append(similar_word)
                            embeddings.append(ftxt[similar_word])
                        
                        embedding_clusters.append(embeddings)
                        word_clusters.append(words)
                    
                    embedding_clusters = np.array(embedding_clusters)
                    n, m, k = embedding_clusters.shape
                    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
                    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                    
                    tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                            f'Sval oko srdce FastText min_count = min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) +', hs = ' + str(activation) + ',  seed = 42.png')    
                        
        else:
            for cbow in cbow_mn:
                for activation in activations:
                    if activation == 0:
                        for negat in negatives:
                                                    
                            words = []
                        
                            for entity in drg_UDPipe:
                                words.append(entity['elements'])
                                    
                            ftxt = FastText(min_count = min_cnt, size = 300, window = 3,  workers = 3, sg = architecture, cbow_mean = cbow, hs = activation, negative = negat, seed = 42)  
                            ftxt.build_vocab(sentences = words)
                            ftxt.train(sentences = words, total_examples = len(words), epochs = 30) 
                        
                            keys = ['sval', 'oko', 'srdce']
                        
                            embedding_clusters = []
                            word_clusters = []
                            
                            for word in keys:
                                embeddings = []
                                words = []
                                for similar_word, _ in ftxt.most_similar(word, topn=30):
                                    words.append(similar_word)
                                    embeddings.append(ftxt[similar_word])
                                embedding_clusters.append(embeddings)
                                word_clusters.append(words)
                            
                            embedding_clusters = np.array(embedding_clusters)
                            n, m, k = embedding_clusters.shape
                            tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
                            embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                            
                            tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                                    f'Sval oko srdce FastText min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) + ', cbow_mean = ' + str(cbow) +', hs = ' + str(activation) + ', negative = '+ str(negat) + ',  seed = 42.png')    
                    else:
                        
                        words = []
                        
                        for entity in drg_UDPipe:
                            words.append(entity['elements'])
                            
                        ftxt= FastText(min_count = min_cnt, size = 300, window = 3,  workers = 3, sg = architecture, cbow_mean = cbow, hs = activation, seed = 42)  
                        ftxt.build_vocab(sentences=words)
                        ftxt.train(sentences=words, total_examples=len(words), epochs=30) 
                    
                        keys = ['sval', 'oko', 'srdce']
                    
                        embedding_clusters = []
                        word_clusters = []
                        
                        for word in keys:
                            embeddings = []
                            words = []
                            
                            for similar_word, _ in ftxt.most_similar(word, topn=30):
                                words.append(similar_word)
                                embeddings.append(ftxt[similar_word])
                           
                            embedding_clusters.append(embeddings)
                            word_clusters.append(words)
                        
                        embedding_clusters = np.array(embedding_clusters)
                        n, m, k = embedding_clusters.shape
                        tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
                        embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                        
                        tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                                f'Sval oko srdce FastText min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) + ', cbow_mean = ' + str(cbow) +', hs = ' + str(activation) + ',  seed = 42.png')    
                            
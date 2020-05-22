import re
import json
import pandas as pd
import dfply as df
import numpy as np

from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from def_udpipe import UDPipe_preprocessing_word
from def_sentence_preprocessing import sentence_preprocessing
from def_word2vec_viz import tsne_plot_similar_words

import logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


def clean_html(raw_html):
  cleanr = re.compile('<.*?>')
  clean_text = re.sub(cleanr, '', raw_html)
  return clean_text

test1 = pd.read_excel('jednotkyZaklad.xlsx')
test1 = test1 >> df.drop(df.contains('Rozsah'), )

test1.columns = ['nazev_jednotky', 'specialista', 'garant', 'sekce', 'lek_disc', 'kurz_kod','kurz_nazov', 'vyznam', 'klic_slova', 'vyzn_pojmy', 'vystup']
test1 = test1.replace(np.nan, "")
test1 = test1 >> df.drop('specialista', 'garant', 'kurz_kod')

# concanating all information into new created column 
test1['all'] = ""
for name_column in test1.columns:
    if name_column == "all":
        pass
    else:
        test1['all'] = test1['all'].str.cat(test1[name_column],  sep=', ')

del name_column

test1 = test1 >> df.drop('nazev_jednotky', 'sekce', 'lek_disc', 'kurz_nazov', 'vyznam', 'klic_slova', 'vyzn_pojmy', 'vystup') #drop all column but "all" 
test1['all'] = test1['all'].str.replace(", ", "", n = 1) #remove first comma 
#------------------------------------------------------------------------------
test2 = pd.read_csv('prispevky.csv')
test2 = test2 >> df.drop('type', 'id', 'category_id')

for i in range(len(test2)):
    test2['body'][i] = clean_html(test2['body'][i])
    
# concanating all information into new created column 
test2['all'] = ""
for name_column in test2.columns:
    if name_column == "all":
        pass
    else:
        test2['all'] = test2['all'].str.cat(test2[name_column],  sep=', ')

del name_column

test2 = test2 >> df.drop('title', 'body', 'category') #drop all column but "all" 
test2['all'] = test2['all'].str.replace(", ", "", n = 1) #remove first comma 

for i in range(len(test2)):
    test2['all'][i] = re.sub('\xa0', " ", test2['all'][i])

test = pd.concat([test1, test2])
#----------------------------------
sum_test1 = 0
for i in range(len(test1)):
    sum_test1 += len(test1['all'][i])
    
2493682 / 1202

sum_test2 = 0
for i in range(len(test2)):
    sum_test2 += len(test2['all'][i])
1289569 / 527  

(sum_test1 + sum_test2)/ 1729  
# ----------------------------------
optimed = [words for words in test['all']] #convert documents into list 
optimed_UDPipe = [UDPipe_preprocessing_word(ls) for ls in optimed] #tokenization + lemmatization 
optimed_preproc = sentence_preprocessing(optimed_UDPipe) #preprocessing


with open('OPTIMED_NZIP_clean.txt', 'w', encoding = 'utf8') as outfile:
    json.dump(optimed_preproc, outfile, ensure_ascii=False)         

with open('OPTIMED_NZIP_clean.txt', 'r', encoding = 'utf8') as file:
    optimed = json.load(file)

sent = [word.split() for word in optimed]

w2v_model= Word2Vec(min_count = 2, size = 50, window = 3,  workers = 1, sg = 1, hs = 0, negative = 5, seed = 0)  
w2v_model.build_vocab(sentences = sent)
w2v_model.train(sentences = sent, total_examples = len(sent), epochs = 30) 

w2v_model.save("word2vec_optimed.model")
w2v_model.save("word2vec_optimed.kv")


#size - the number of dimension of the embeddings, def = 100
#window - the maximum distance between a target word and words around the target word, def = 5
#min_count - the minimum count of words to consider when training the model; words with occurrence less than this count will be ignored, def = 5
#workers - the number of partitions during training , def = 3
#sg - the training algorithm, CBOW - 0, skipgram - 1, defaul = CBOW

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
                        
                                
                        w2v_model= Word2Vec(min_count = min_cnt, size = 50, window = 3,  workers = 3, sg = architecture, hs = activation, negative = negat, seed = 0)  
                        w2v_model.build_vocab(sentences = sent)
                        w2v_model.train(sentences = sent, total_examples = len(sent), epochs = 30) 
                    
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
                        tsne_model_en_2d = TSNE(perplexity = 15, n_components = 2, init = 'pca', n_iter = 3500, random_state = 32)
                        embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                        
                        tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                                f'Sval oko srdce OPTIMED NZIP min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) +', hs = ' + str(activation) + ', negative = '+ str(negat) + ',  seed = 0.png')    
                else:
                    w2v_model= Word2Vec(min_count = min_cnt, size = 50, window = 3,  workers = 3, sg = architecture, hs = activation, seed = 0)  
                    w2v_model.build_vocab(sentences = sent)
                    w2v_model.train(sentences = sent, total_examples=len(sent), epochs = 30) 
                
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
                    tsne_model_en_2d = TSNE(perplexity = 15, n_components = 2, init = 'pca', n_iter = 3500, random_state = 32)
                    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                    
                    tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                            f'Sval oko srdce OPTIMED NZIP min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) +', hs = ' + str(activation) + ',  seed = 0.png')    
                        
        else:
            for cbow in cbow_mn:
                for activation in activations:
                    if activation == 0:
                        for negat in negatives:
                                                    

                                    
                            w2v_model= Word2Vec(min_count = min_cnt, size = 50, window = 3,  workers = 1, sg = architecture, cbow_mean = cbow, hs = activation, negative = negat, seed = 0)  
                            w2v_model.build_vocab(sentences = sent)
                            w2v_model.train(sentences = sent, total_examples=len(sent), epochs=30) 
                        
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
                            tsne_model_en_2d = TSNE(perplexity = 15, n_components = 2, init = 'pca', n_iter = 3500, random_state = 32)
                            embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                            
                            tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                                    f'Sval oko srdce OPTIMED NZIP min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) + ', cbow_mean = ' + str(cbow) +', hs = ' + str(activation) + ', negative = '+ str(negat) + ',  seed = 0.png')    
                    else:
                        w2v_model= Word2Vec(min_count = min_cnt, size = 50, window = 3,  workers = 1, sg = architecture, cbow_mean = cbow, hs = activation, seed = 0)  
                        w2v_model.build_vocab(sentences = sent)
                        w2v_model.train(sentences = sent, total_examples=len(sent), epochs=30) 
                    
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
                        tsne_model_en_2d = TSNE(perplexity = 15, n_components = 2, init = 'pca', n_iter = 3500, random_state = 32)
                        embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                        
                        tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                                f'Sval oko srdce OPTIMED NZIP min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) + ', cbow_mean = ' + str(cbow) +', hs = ' + str(activation) + ',  seed = 0.png')    
                            
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
                                
                        ftxt= FastText(min_count = min_cnt, size = 300, window = 3,  workers = 3, sg = architecture, hs = activation, negative = negat, seed = 0)  
                        ftxt.build_vocab(sentences = sent)
                        ftxt.train(sentences = sent, total_examples = len(sent), epochs = 30) 
                    
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
                        tsne_model_en_2d = TSNE(perplexity = 15, n_components = 2, init = 'pca', n_iter = 3500, random_state = 32)
                        embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                        
                        tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                                f'Sval oko srdce FastText OPTIMED NZIP min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) +', hs = ' + str(activation) + ', negative = '+ str(negat) + ',  seed = 0.png')    
                else:
                        
                    ftxt= FastText(min_count = min_cnt, size = 300, window = 3,  workers = 3, sg = architecture, hs = activation, seed = 0)  
                    ftxt.build_vocab(sentences = sent)
                    ftxt.train(sentences = sent, total_examples = len(sent), epochs = 30) 
                
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
                    tsne_model_en_2d = TSNE(perplexity = 15, n_components = 2, init = 'pca', n_iter = 3500, random_state = 32)
                    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                    
                    tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                            f'Sval oko srdce FastText OPTIMED NZIP min_count = min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) +', hs = ' + str(activation) + ',  seed = 0.png')    
                        
        else:
            for cbow in cbow_mn:
                for activation in activations:
                    if activation == 0:
                        for negat in negatives:
                                    
                            ftxt = FastText(min_count = min_cnt, size = 300, window = 3,  workers = 3, sg = architecture, cbow_mean = cbow, hs = activation, negative = negat, seed = 0)  
                            ftxt.build_vocab(sentences = sent)
                            ftxt.train(sentences = sent, total_examples = len(sent), epochs = 30) 
                        
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
                            tsne_model_en_2d = TSNE(perplexity = 15, n_components = 2, init = 'pca', n_iter = 3500, random_state = 32)
                            embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                            
                            tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                                    f'Sval oko srdce FastText OPTIMED NZIP min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) + ', cbow_mean = ' + str(cbow) +', hs = ' + str(activation) + ', negative = '+ str(negat) + ',  seed = 0.png')    
                    else:
                        
                            
                        ftxt= FastText(min_count = min_cnt, size = 300, window = 3,  workers = 3, sg = architecture, cbow_mean = cbow, hs = activation, seed = 0)  
                        ftxt.build_vocab(sentences = sent)
                        ftxt.train(sentences = sent, total_examples=len(sent), epochs=30) 
                    
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
                        tsne_model_en_2d = TSNE(perplexity = 15, n_components = 2, init = 'pca', n_iter = 3500, random_state = 32)
                        embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                        
                        tsne_plot_similar_words('Sval oko srdce', keys, embeddings_en_2d, word_clusters, 0.7,
                                                f'Sval oko srdce FastText OPTIMED NZIP min_count = ' + str(min_cnt) + ' , size = 300, workers = 3, window = 3, sg = ' + str(architecture) + ', cbow_mean = ' + str(cbow) +', hs = ' + str(activation) + ',  seed = 0.png')    

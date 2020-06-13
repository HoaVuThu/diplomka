import re
import json
import codecs
import numpy as np
import pandas as pd
from lxml import html
from stop_words import get_stop_words
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#------------------------------------------------------------------------------
# Xpath extraction
def clean_data(data, xpath):

#loading data and Xpaths
    jsonData = codecs.open(data, 'r', 'utf-8-sig').read()
    data = json.loads(jsonData)
    result = []
    excelData = pd.read_excel(xpath)
    xpaths = pd.DataFrame(excelData, columns = ['Xpath'], dtype = str)        

#cleaning data from html tags 
    for entity in data:
        htmlString = entity['html']
        elements = []
        tree = html.fromstring(htmlString)
        for lxpath in xpaths['Xpath']:
            elements.append(tree.xpath(lxpath))
        result.append({"elements": elements, "url": entity["url"]})

#remove whitespace      
    for entity in result:
        for onelist in entity['elements']:
             for i in range(len(onelist)):
                 onelist[i] = onelist[i].translate({ord('\n'): None}).translate({ord('\t'): None}).translate({ord('•'):None})

#creating new feature - category - opravit, dava vsade []
    for entity in result:
        if entity['elements'][2]  == "":
            entity['category'] = "jiné"
        else:
            entity['category'] = entity['elements'][2] #treba spravit tu tabulku aby dobre vytahovalo a potom spravit kategoriu 

#removing empty lists and strings and concanate all list into one document 
    for entity in result:
        entity['elements'] = sum(entity['elements'], [])
        entity['elements'] = [x for x in entity['elements'] if x] 
        
    return result

#------------------------------------------------------------------------------
# filtering out HTML tags

def clean_html(raw_html):
  cleanr = re.compile('<.*?>')
  clean_text = re.sub(cleanr, '', raw_html)
  return clean_text

#------------------------------------------------------------------------------
# sentence preprocessing

def sentence_preprocessing(text): 
    
    digits = '0123456789'
    spec_char = [".", "," ,";", "-", "(", ")", "[", "]", '"\"', "/", "_"] 
    stop = get_stop_words('czech')
    stop2 = pd.read_excel('stop.xlsx')
    stop += stop2['stop_words'].values.tolist()
    remove_digits = str.maketrans('', '', digits)
        
    result = []
    
    for ls in text:
        # ls = re.sub(r'^[0-9][0-9](.*)?[0-9][0-9]+$' ,'', ls)#remove DRG codes 
        words = re.split('\W+', ls) #split sentence and abbreviations (for example: zpus.j.vnitr.)
        tmp_sentence = []
        for word_ls in words:
            word = word_ls.translate(remove_digits) #remove digits
            word = re.sub(r"\W+", '', word) #remove word from special characters           
            word = word.lower() #lower words 
            
            if len(word) == 1:
                continue
            if word in stop or word in spec_char:  #remove words from stop words list
                continue
              
            tmp_sentence.append(word)
            
        tmp_res = ' '.join(tmp_sentence)
        result.append(tmp_res)
 
    return result  

#------------------------------------------------------------------------------
# visualization 

def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()



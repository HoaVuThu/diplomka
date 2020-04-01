from stop_words import get_stop_words
import re 
import pandas as pd

def sentence_preprocessing(text): 
    
    digits = '0123456789'
    spec_char = [".", "," ,";", "-", "(", ")", "[", "]", '"\"', "/", " "] 
    stop = get_stop_words('czech')
    stop2 = pd.read_excel('stop.xlsx')
    stop += stop2['stop_words'].values.tolist()
    remove_digits = str.maketrans('', '', digits)
        
    result = []
    
    for ls in text:
        ls = re.sub(r'^[0-9][0-9](.*)?[0-9][0-9]+$' ,'', ls)#remove DRG codes 
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
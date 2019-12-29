from stop_words import get_stop_words
import re 
from string import digits

spec_char = [".", "," ,";", "-", "(", ")", "[", "]", '"\"', "/", " "] 
stop = get_stop_words('czech')
for i in ['a', 'aj', 's', 'při', 'k', 'v', 'o', 'z', 'i', 'u','či' ,'ze', 'ke', 'do', 'po', 'se','aby', 'až', 'ať', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX','X', 'alespoň']:
    stop.append(i)
remove_digits = str.maketrans('', '', digits)

def sentence_preprocessing(text): 
        
    result = []
    
    for ls in text:
        ls = ls.replace('_', ' ') #remove '_'
        ls = ls.translate(remove_digits) #remove digits
        words = re.split('\W+', ls) #split sentence and abbreviations 
        tmp_sentence = []
        for word_ls in words:
            word = re.sub(r"\W+", '', word_ls) #remove word from special characters       
            
            if word in stop or word in spec_char :  #remove words from stop words list
                continue
            
            if word.isupper(): #keep abbreviations as HIV, AIDS 
                tmp_sentence.append(word) 
            
            else:             
                word = word.lower() #lower words 
                tmp_sentence.append(word)
            
        tmp_res = ' '.join(tmp_sentence)
        result.append(tmp_res)
 
    return result
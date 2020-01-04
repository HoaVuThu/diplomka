from stop_words import get_stop_words
import re 
from string import digits

spec_char = [".", "," ,";", "-", "(", ")", "[", "]", '"\"', "/", " "] 
stop = get_stop_words('czech')
for i in ['a', 'aj', 's', 'při', 'k', 'v', 'o', 'z', 'i', 'u','či' ,'ze', 'ke', 'do', 'po', 'se','aby', 'až', 'ať', 'I', 'II', 'III']:
    stop.append(i)
remove_digits = str.maketrans('', '', digits)

def word_preprocessing(text):
            
    result = []
    
    for ls in text:
        #split abbreviations and words
        words = ls.split()
        for word_ls in words:
            #remove word from special characters 
            word = re.sub(r"\W+", '', word_ls)       
            #remove words from stop words list 
            if word in stop or word in spec_char:  
                continue
            if word.isupper(): #keep abbreviations as HIV, AIDS 
                result.append(word) 
            
            else:             
                word = word.lower() #lower words 
                result.append(word)
 
    return result
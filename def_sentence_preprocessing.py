from stop_words import get_stop_words
import re 

def sentence_preprocessing(text): 
    
    digits = '0123456798'
    spec_char = [".", "," ,";", "-", "(", ")", "[", "]", '"\"', "/", " "] 
    stop = get_stop_words('czech')
    for i in ['a', 'aj', 's', 'při', 'k', 'v', 'o', 'z', 'i', 'u','či' ,'ze', 'ke', 'do', 'po', 'se','aby', 'až', 'ať', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX','X', 'alespoň', 'nad', 'pod', 'ZZ', 'CC']:
        stop.append(i)
    remove_digits = str.maketrans('', '', digits)
        
    result = []
    
    for ls in text:
        words = re.split('\W+', ls) #split sentence and abbreviations (for example: zpus.j.vnitr.)
        tmp_sentence = []
        for word_ls in words:
            #word = re.sub(r'^[0-9][0-9](.*)?[0-9][0-9]+$' ,'', word_ls)#remove DRG codes 
            word = word_ls.translate(remove_digits) #remove digits
            word = re.sub(r"\W+", '', word) #remove word from special characters           
            
            if len(word) == 1:
                continue
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
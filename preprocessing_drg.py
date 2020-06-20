import json

from def_functions import sentence_preprocessing, clean_data
from def_udpipe import UDPipe_preprocessing_word

### DRG

## data preprocessing
drg = clean_data('.\\drg\\drg-without-comma.json', '.\\drg\\DRG-klasifikace-XPath.xlsx')

# lemmatization    
for entity in drg:
    entity['elements'] = [UDPipe_preprocessing_word(ls) for ls in entity['elements']]

# filtering
for entity in drg:
    entity['elements'] = sentence_preprocessing(entity['elements'])# 1 908 documents

for entity in drg:
    entity['elements'] = " ".join(entity['elements'])

for entity in drg:
    entity['elements'] = list(entity['elements'].split())
        
    
with open('drg_clean.txt', 'w', encoding = 'utf8') as outfile:
    json.dump(drg, outfile, ensure_ascii=False)         

    
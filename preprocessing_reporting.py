import json
import codecs
import pandas as pd
from lxml import html
import re
from def_udpipe import UDPipe_preprocessing_word
from def_sentence_preprocessing import sentence_preprocessing
from def_remove_HTML_tags import clean_html

jsonData = codecs.open('.\\reporting\\reporting-without-comma.json', 'r', 'utf-8-sig').read()
reporting = json.loads(jsonData)
    
for entity in reporting:
    entity['elements'] = entity['html'][entity['html'].find("<body>") : entity['html'].find("</body>")]  
    entity['elements'] = clean_html(entity['elements'])
    entity['elements'] = entity['elements'][: entity['elements'].find("Váš komentář")] 
    entity['elements'] = entity['elements'][: entity['elements'].find("Tento web používá k poskytování služeb a analýze návštěvnosti soubory cookie")] 
    entity['elements'] = entity['elements'].replace('&emsp;', "")
    entity['elements'] = entity['elements'].replace("\n", " ").replace('\t', " ").replace("  ", " ")
    entity['elements'] = re.sub('[$].*;', "", entity['elements'])


for i, entity in enumerate(reporting):
    print(f'UDPIPE processing list number: ' + str(i))
    entity['elements'] = [UDPipe_preprocessing_word(entity['elements'])]
    print('done')
    
for i, entity in enumerate(reporting):
    print(f'processing list number: ' + str(i))
    entity['elements'] = sentence_preprocessing(entity['elements'])
    print('done')

# remove extra space    
for entity in reporting:
    for i in range(len(entity['elements'])):
        entity['elements'][i] = re.sub('  ', ' ', entity['elements'][i], flags = re.UNICODE)        

for entity in reporting:
    entity['elements'] = " ".join(entity['elements'])

for entity in reporting:
    entity['elements'] = list(entity['elements'].split())
    
with open('reporting_clean.txt', 'w', encoding = 'utf8') as outfile:
    json.dump(reporting, outfile, ensure_ascii = False)  #udpipe + sentence preprossing 
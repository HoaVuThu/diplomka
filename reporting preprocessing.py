import json
import codecs
import pandas as pd
from lxml import html
import re

jsonData = codecs.open('.\\reporting\\reporting-without-comma.json', 'r', 'utf-8-sig').read()
reporting = json.loads(jsonData)

def clean_html(raw_html):
  cleanr = re.compile('<.*?>')
  clean_text = re.sub(cleanr, '', raw_html)
  clean_text = re.sub('^')
  return clean_text
    
for entity in reporting:
    entity['elements'] = entity['html'][entity['html'].find("<body>") : entity['html'].find("</body>")]  
    entity['elements'] = clean_html(entity['elements'])
    entity['elements'] = entity['elements'][: entity['elements'].find("   Váš komentář")]  

#print(reporting[100]['elements'])

for entity in reporting:
    entity['elements'] = [UDPipe_preprocessing_word(ls) for ls in entity['elements']]

for i, entity in enumerate(reporting):
    print(f'processing list number: ' + str(i))
    entity['elements'] = sentence_preprocessing(entity['elements'])
    print('done')
 
with open('reporting_clean.txt', 'w', encoding = 'utf8') as outfile:
    json.dump(reporting, outfile, ensure_ascii=False) #udpipe            

with open('reporting_clean.txt', 'r', encoding = 'utf8') as file:
    reporting = json.load(file)
    



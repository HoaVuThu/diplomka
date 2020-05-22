import json
import codecs
import pandas as pd
from lxml import html
import re
from def_udpipe import UDPipe_preprocessing_word
from def_sentence_preprocessing import sentence_preprocessing

jsonData = codecs.open('.\\reporting\\reporting-without-comma.json', 'r', 'utf-8-sig').read()
reporting3 = json.loads(jsonData)

def clean_html(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
#  cleantext = re.sub('^')
  return cleantext
    
for entity in reporting3:
    entity['elements'] = entity['html'][entity['html'].find("<body>") : entity['html'].find("</body>")]  
    entity['elements'] = clean_html(entity['elements'])
    entity['elements'] = entity['elements'][: entity['elements'].find("   Váš komentář")] 
    entity['elements'] = entity['elements'][: entity['elements'].find("   Tento web používá k poskytování služeb a analýze návštěvnosti soubory cookie")] 
    entity['elements'] = entity['elements'].replace('&emsp;', "")
    entity['elements'] = entity['elements'].replace("\n", " ").replace('\t', " ").replace("  ", " ")
    entity['elements'] = re.sub('[$].*;', "", entity['elements'])


print(reporting[510]['elements'])
print(reporting2[510]['elements'])

print(reporting[1000]['elements'])
print(reporting2[1000]['elements'])
print(reporting3[1000]['elements'])
print(reporting3[10000]['elements'])
print(reporting3[784]['elements'])
print(reporting3[1784]['elements'])
print(reporting_orig[1784]['elements'])




for i, entity in enumerate(reporting3):
    print(f'UDPIPE processing list number: ' + str(i))
    entity['elements'] = [UDPipe_preprocessing_word(entity['elements'])]
    print('done')

with open('reporting_clean.txt', 'w', encoding = 'utf8') as outfile:
    json.dump(reporting2, outfile, ensure_ascii=False) #udpipe 
    
for i, entity in enumerate(reporting2):
    print(f'processing list number: ' + str(i))
    entity['elements'] = sentence_preprocessing(entity['elements'])
    print('done')
 
with open('reporting_clean_2.txt', 'w', encoding = 'utf8') as outfile:
    json.dump(reporting2, outfile, ensure_ascii=False)  #udpipe + sentence prepros           

with open('reporting_clean.txt', 'r', encoding = 'utf8') as file:
    reporting = json.load(file)
    

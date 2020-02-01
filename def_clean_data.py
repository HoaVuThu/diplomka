import json
import codecs
import pandas as pd
from lxml import html

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

    


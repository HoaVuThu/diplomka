import pandas as pd
import dfply as df
import numpy as np

from def_functions import clean_html, sentence_preprocessing 
from def_udpipe import UDPipe_preprocessing_word

#------------------------------------------------------------------------------
### Prepare data from OPTIMED 
optimed = pd.read_excel('jednotkyZaklad.xlsx')
optimed = optimed >> df.drop(df.contains('Rozsah'), )

optimed.columns = ['nazev_jednotky', 'specialista', 'garant', 'sekce', 'lek_disc', 'kurz_kod','kurz_nazov', 'vyznam', 'klic_slova', 'vyzn_pojmy', 'vystup']
optimed = optimed.replace(np.nan, "")
optimed = optimed >> df.drop('specialista', 'garant', 'kurz_kod')

optimed['doc1'] = ""

#concenate all columns into one 

for name_column in optimed.columns:
    if name_column == 'lek_disc' or name_column == "doc1":
        pass
    else:
        optimed['doc1'] = optimed['doc1'].str.cat(optimed[name_column],  sep=', ')
        
optimed['doc1'] = optimed['doc1'].str.replace(", ", "", n = 1)

#------------------------------------------------------------------------------
# create cca balanced dataset
optimed_sorted = optimed.iloc[0:len(optimed)//2, :]
optimed_not_sorted = optimed.iloc[len(optimed)//2:, :]

optimed_sorted = pd.DataFrame(optimed_sorted, columns = ['lek_disc', 'doc1']).sort_values(by = ["lek_disc"])
optimed_not_sorted = optimed_not_sorted[['lek_disc','doc1']]

optimed_sorted_0 = optimed_sorted.iloc[0:len(optimed_sorted):2, :].reset_index()
optimed_sorted_1 = optimed_sorted.iloc[1:len(optimed_sorted):2, :].reset_index()
optimed_sorted = pd.concat([optimed_sorted_0, optimed_sorted_1], axis = 1, ignore_index = False)
optimed_sorted .columns = ['index1','lek_disc1', 'doc1', 'index2','lek_disc2', 'doc2']


optimed_not_sorted_0 = optimed_not_sorted.iloc[0:len(optimed_not_sorted):2, :].reset_index()
optimed_not_sorted_1 = optimed_not_sorted.iloc[1:len(optimed_not_sorted):2, :].reset_index()
optimed_not_sorted = pd.concat([optimed_not_sorted_0, optimed_not_sorted_1], axis = 1, ignore_index = False)
optimed_not_sorted .columns = ['index1','lek_disc1', 'doc1', 'index2','lek_disc2', 'doc2']


new_optimed = optimed_sorted.append(optimed_not_sorted).reset_index()
new_optimed = new_optimed >> df.drop(df.contains('index'))
#------------------------------------------------------------------------------
#create class: 0 - not similar , 1 - similar

new_optimed['similar'] = 0

for i in range(len(new_optimed)):
    print(i)
    if new_optimed.lek_disc1[i] == new_optimed.lek_disc2[i]:
        new_optimed['similar'][i] = 1


new_optimed = new_optimed >> df.drop(df.contains('lek'))

new_optimed ['similar'].value_counts()
# -----------------------------------------------------------------------------
#------------------------------------------------------------------------------
### Prepare data from NZIP

nzip = pd.read_csv('prispevky.csv')
nzip = nzip >> df.drop('type', 'id', 'category_id')

# Remove HTML tags
for i in range(len(nzip)):
    nzip['body'][i] = clean_html(nzip['body'][i])
    

nzip['doc1'] = ""

# concenate all columns into one 
for name_column in nzip.columns:
    if name_column == 'category' or name_column == "doc1":
        pass
    else:
        nzip['doc1'] = nzip.doc1.str.cat(nzip[name_column],  sep=', ')


nzip = nzip >> df.drop('title', 'body')  
nzip['doc1'] = nzip['doc1'].str.replace(", ", "", n = 1) #remove first comma 

nzip = pd.DataFrame(nzip, columns = ['category', 'doc1'])

# -----------------------------------------------------------------------------

# # create cca balanced dataset, every documents used once 
nzip_sorted = nzip.iloc[0:len(nzip)//2, :]
nzip_not_sorted = nzip.iloc[len(nzip)//2:, :]

nzip_sorted = pd.DataFrame(nzip_sorted, columns = ['category', 'doc1']).sort_values(by = ["category"])

nzip_sorted_0 = nzip_sorted.iloc[0:len(nzip_sorted):2, :].reset_index()
nzip_sorted_1 = nzip_sorted.iloc[1:len(nzip_sorted):2, :].reset_index()
nzip_sorted = pd.concat([nzip_sorted_0, nzip_sorted_1], axis = 1, ignore_index = False)
nzip_sorted .columns = ['index1','category1', 'doc1', 'index2','category2', 'doc2']

nzip_not_sorted_0 = nzip_not_sorted.iloc[0:len(nzip_not_sorted):2, :].reset_index()
nzip_not_sorted_1 = nzip_not_sorted.iloc[1:len(nzip_not_sorted):2, :].reset_index()
nzip_not_sorted = pd.concat([nzip_not_sorted_0, nzip_not_sorted_1], axis = 1, ignore_index = False)
nzip_not_sorted .columns = ['index1','category1', 'doc1', 'index2','category2', 'doc2']
       
new_nzip= nzip_sorted.append(nzip_not_sorted).reset_index()
new_nzip = new_nzip >> df.drop(df.contains('index'))

#------------------------------------------------------------------------------
#create class: 0 - not similar , 1 - similar

new_nzip['similar'] = 0

for i in range(len(new_nzip)):
    print(i)
    if new_nzip.category1[i] == new_nzip.category2[i]:
        new_nzip['similar'][i] = 1

new_nzip = new_nzip >> df.drop(df.contains('category'))

new_nzip['similar'].value_counts()

#------------------------------------------------------------------------------
# concenate NZIP and OPTIMED and pre-processing 
data = new_nzip.append(new_optimed).reset_index()
data['similar'].value_counts() # count in every class 

data['doc1'] = data['doc1'].astype('str') 
data['doc2'] = data['doc2'].astype('str')

data_UDPipe = data
data_UDPipe['doc1'] = [UDPipe_preprocessing_word(ls) for ls in data['doc1']] #tokenization + lemmatization 
data_UDPipe['doc2'] = [UDPipe_preprocessing_word(ls) for ls in data['doc2']] #tokenization + lemmatization 
data_preproc = data
data_preproc['doc1'] = sentence_preprocessing(data_UDPipe['doc1']) #preprocessing
data_preproc['doc2'] = sentence_preprocessing(data_UDPipe['doc2'])

data.to_csv (r'MALSTM_train_data.csv', index = None, header=True) 
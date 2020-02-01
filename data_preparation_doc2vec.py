import pandas as pd
import dfply as df
import numpy as np

test = pd.read_excel('jednotkyZaklad.xlsx')
test = test >> df.drop(df.contains('Rozsah'), )

test.columns = ['nazev_jednotky', 'specialista', 'garant', 'sekce', 'lek_disc', 'kurz_kod','kurz_nazov', 'vyznam', 'klic_slova', 'vyzn_pojmy', 'vystup']
test = test.replace(np.nan, "")
test = test >> df.drop('specialista', 'garant', 'kurz_kod')

test['doc1'] = ""

for name_column in test.columns:
    if name_column == 'lek_disc' or name_column == "doc1":
        pass
    else:
        test['doc1'] = test.doc1.str.cat(test[name_column],  sep=', ')
        
test.doc1 = test['doc1'].str.replace(", ", "", n = 1)

test = pd.DataFrame(test, columns = ['lek_disc', 'doc1'])

new_test1 = test.iloc[0:len(test):2, :].reset_index()
new_test2 = test.iloc[1:len(test):2, :].reset_index()
new_test = pd.concat([new_test1, new_test2], axis = 1, ignore_index = False)

new_test.columns = ['index1','lek_disc1', 'doc1', 'index2','lek_disc2', 'doc2']

new_test['similar'] = 0

for i in range(len(new_test)):
    if new_test.lek_disc1[i] == new_test.lek_disc2[i]:
        new_test['similar'][i] = 1

new_test = new_test >> df.drop(df.contains('index'), df.contains('lek'))

test = new_test.to_csv (r'test.csv', index = None, header=True) 

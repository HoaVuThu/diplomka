libraries:  ufal.udpipe, stop_words, re, pandas, gensim, smart_open, json 

download and move into one folder:
  cmdParamReadingExample.py
  def_udpipe.py
  def_sentence_preprocessing.py 
  doc2vec_optimed.model
  
from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131 download "model czech-pdt-ud-2.5-191206.udpipe"

change path in def_udpipe to path where you downloaded "czech-pdt-ud-2.5-191206.udpipe"

oped cmd and change directory to the one where you moved downloaded python skripts 

write: python cmdParamReadingExample.py "sentence 1" "sentence 2"

output:
preprocessed sentence 1
preprocessed sentence 2
cosine similarity
ranged cosine similarity 

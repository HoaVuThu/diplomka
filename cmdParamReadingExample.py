import sys
from gensim.models import doc2vec
from scipy import spatial

from def_udpipe import UDPipe_preprocessing_word
from def_sentence_preprocessing import sentence_preprocessing

firstArg = sys.argv[1]
secondArg = sys.argv[2]


preproc_1 = sentence_preprocessing([firstArg])
preproc_2 = sentence_preprocessing([secondArg])

UDPipe_1 = [UDPipe_preprocessing_word(ls) for ls in preproc_1]
UDPipe_2 = [UDPipe_preprocessing_word(ls) for ls in preproc_2]

str_1 = " ".join(UDPipe_1)
str_2 = " ".join(UDPipe_2)

### doc2vec

model = doc2vec.Doc2Vec.load('doc2vec_optimed.model')

vec1 =  model.infer_vector(UDPipe_1)
vec2 =  model.infer_vector(UDPipe_2)

cos = 1 - spatial.distance.cosine(vec1, vec2)
similarity = round(((cos + 1)/2)*100, 2) 

print ("First Argument: " + str_1)
print ("Second Argument: " + str_2)
print ("Cosinus sim: " + str(cos))
print (similarity)
import sys
from gensim.models import doc2vec
from scipy import spatial
from def_udpipe import UDPipe_preprocessing_word
from def_sentence_preprocessing import sentence_preprocessing
from def_remove_HTML_tags import clean_html


Arg_1 = sys.argv[1]
Arg_2 = sys.argv[2]

cleaned_Arg_1 = clean_html(Arg_1)
cleaned_Arg_2  = clean_html(Arg_2)

UDPipe_1 = [UDPipe_preprocessing_word(cleaned_Arg_1)]
UDPipe_2 = [UDPipe_preprocessing_word(cleaned_Arg_2)]

preproc_1 = sentence_preprocessing(UDPipe_1)
preproc_2 = sentence_preprocessing(UDPipe_2)

str_1 = " ".join(preproc_1)
str_2 = " ".join(preproc_2)

sentence_1 = str_1.split()
sentence_2 = str_2.split()
### doc2vec
model = doc2vec.Doc2Vec.load('doc2vec_optimed_NZIP.model')

model.random.seed(0)
vec1 =  model.infer_vector(sentence_1)
vec2 =  model.infer_vector(sentence_2)

cos = 1 - spatial.distance.cosine(vec1, vec2)
#similarity = round(((cos + 1)/2)*100, 2) 

print("First Argument: " + str_1)
print("Second Argument: " + str_2)
print("Cosine similarity: " + str(cos))
#print("Cosinus similarity ranged from 0 to 1: " + str(similarity))
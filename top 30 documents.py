from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
import smart_open
import json
import random
from scipy import spatial 
import pandas as pd 

def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])
        
with open('OPTIMED_clean_preprocessed.txt', 'r', encoding = 'utf8') as file:
    train_corpus = json.load(file)

train_corpus= [list(ls.split(" ")) for ls in train_corpus]
 

train_corpus = list(create_tagged_document(train_corpus))

with open('drg_clean_preprocessed_json.txt', 'r', encoding = 'utf8') as file:
    test_corpus = json.load(file)


model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

model.init_sims(replace=True)
model.save("doc2vec_optimed.model")
model.save("doc2vec_optimed.kv")
#########

sub_test_corpus = test_corpus[-31:-1]
 
output = pd.DataFrame(columns = ["doc1", "doc2", "link_doc1", "link_doc2" ,"cosine_similarity", "ranged_similarity"])

n = len(sub_test_corpus)
for i, entity_i in enumerate(sub_test_corpus):
    if i == n:
        break
    for j, entity_j in enumerate(sub_test_corpus):
        if j >= i:
            continue
        vec1 = model.infer_vector(entity_i['elements'])
        vec2 = model.infer_vector(entity_j['elements'])
        cos = 1 - spatial.distance.cosine(vec1, vec2)
        similarity = round(((cos + 1)/2)*100, 2)
        output = output.append({'doc1': i, 'doc2': j, 'link_doc1': entity_i['url'], 'link_doc2': entity_j['url'] ,'cosine_similarity': cos, 'ranged_similarity': similarity}, ignore_index=True)
        
final = output.sort_values(by = ['ranged_similarity'])[:10]
final = final.append(output.sort_values(by = ['ranged_similarity'], ascending = False)[:10])
final = final.append(output.sort_values(by = ['ranged_similarity'], ascending = False)[(len(output) // 2) - 5 : (len(output) // 2) + 5 ])

for i in range(n):
    with open('doc'+str(i)+'.txt', 'w', encoding = 'utf8') as outfile:
        json.dump(sub_test_corpus[i], outfile, ensure_ascii=False)      
        
final.to_csv('output.csv', sep='\t', encoding='utf-8')

import gensim
import json
import random
from scipy import spatial 
import pandas as pd 


def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])
        
with open('OPTIMED_NZIP_clean.txt', 'r', encoding = 'utf8') as file:
    train_corpus = json.load(file)

train_corpus= [list(ls.split(" ")) for ls in train_corpus]
 

train_corpus = list(create_tagged_document(train_corpus))

with open('drg_clean.txt', 'r', encoding = 'utf8') as file:
    test_corpus_json = json.load(file)
    
test_corpus = []
for entity in test_corpus_json:
#    text.append(entity['elements'])
    test_corpus.append(entity['elements'])

model = gensim.models.doc2vec.Doc2Vec(vector_size = 50, min_count = 2, window = 3, epochs = 30, workers = 1, dm = 0, dbow_words = 1, seed = 0)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples = model.corpus_count, epochs = model.epochs)

ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    second_ranks.append(sims[1])

doc_id = random.randint(0, len(train_corpus) - 1)
# Compare and print the second-most-similar document
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
sim_id = second_ranks[doc_id]
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))

# save model
model.save('doc2vec_optimed_NZIP.model')

# -----------------------------------------------------------------------------
#testing

#load datasets to test
with open('drg_clean.txt', 'r', encoding = 'utf8') as file:
    test_drg = json.load(file)
    
with open('reporting_clean.txt', 'r', encoding = 'utf8') as file:
    test_reporting_all = json.load(file)

test_reporting = random.sample(test_reporting_all, 200)

#load doc2vec model
model = gensim.models.doc2vec.Doc2Vec.load('doc2vec_optimed_NZIP.model')
 
#creating and empty dataframe with pre-defined columns 
output = pd.DataFrame(columns = ["index_doc_reporting", "index_doc_drg", "text_reporting", "text_drg", "link_doc_reporting", "link_doc_drg" ,"cosine_similarity", "ranged_similarity"])

#compute similarity between documents from reporting and drg
n_drg = len(test_drg)
n_reporting = len(test_reporting)

for i_reporting, entity_reporting in enumerate(test_reporting):
    print(f'Reporting list numner: ' + str(i_reporting))

    for j_drg, entity_drg in enumerate(test_drg):
       # print(f'DRG list numner: ' + str(j_drg))
        
        vec1 = model.infer_vector(entity_reporting['elements'])
        vec2 = model.infer_vector(entity_drg['elements'])
        cos = 1 - spatial.distance.cosine(vec1, vec2)
        similarity = round(((cos + 1)/2)*100, 2)
        output = output.append({'index_doc_reporting': i_reporting, 'index_doc_drg': j_drg, "text_reporting": entity_reporting['elements'], "text_drg": entity_drg['elements'], 'link_doc_reporting': entity_reporting['url'], 'link_doc_drg': entity_drg['url'] ,'cosine_similarity': cos, 'ranged_similarity': similarity}, ignore_index=True)

#get 10 most similar document, 10 most dissimilar documents and 10 mediocre documents      
final = output.sort_values(by = ['ranged_similarity'])[:20]
final = final.append(output.sort_values(by = ['ranged_similarity'], ascending = False)[:20])
final = final.append(output.sort_values(by = ['ranged_similarity'], ascending = False)[(len(output) // 2) - 10 : (len(output) // 2) + 10 ])
        
final.to_csv('output.csv', sep='\t', encoding='utf-8')
output.to_csv('output_all.csv', sep='\t', encoding='utf-8')
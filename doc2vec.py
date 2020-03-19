from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
import smart_open
import json
import random

def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])
        
with open('OPTIMED_clean_preprocessed.txt', 'r', encoding = 'utf8') as file:
    train_corpus = json.load(file)

train_corpus= [list(ls.split(" ")) for ls in train_corpus]
 

train_corpus = list(create_tagged_document(train_corpus))

with open('drg_clean_preprocessed.txt', 'r', encoding = 'utf8') as file:
    test_corpus = json.load(file)

print(train_corpus[:2])
print(test_corpus[:2])
sub_test_corpus = test_corpus[-31:-1]

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

#save model 


ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])

print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
    
import random
doc_id = random.randint(0, len(train_corpus) - 1)

# Compare and print the second-most-similar document
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
sim_id = second_ranks[doc_id]
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))

# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(test_corpus) - 1)
doc_id = 1363
inferred_vector = model.infer_vector(test_corpus[doc_id])
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
    
    #########
    
    
from gensim.models import doc2vec
from scipy import spatial

#load model
#model = doc2vec.Doc2Vec.load(model_file)

doc_id_train = random.randint(0, len(train_corpus) - 1)
vec1 = model.infer_vector("balintovský skupina diagnostický obor neurověda komunikace sebezkušenost komunikace sebezkušenost cvičení student projít zkušenost bálintovský skupina jakožto efektivní preventivní metoda vůči syndrom vyhoření osobnost praxe psychologie životní zkušenost bálintovský skupina korekce postoj řešení konflikt spor získání náhled charakter vztah zpětný vazba student charakterizovat orientace vztah postoj zpětný vazba korekce postoj".split())
vec1 = model.infer_vector(["srdecni nemoc", "infarkt"])
vec1 = model.infer_vector(["plicni nemoc", "dychani"])


#0.60967 docID test 1363
doc_id_test = random.randint(0, len(test_corpus) - 1)
doc_id_test = 1363
vec2 = model.infer_vector(test_corpus[doc_id_test])

similairty = 1 - spatial.distance.cosine(vec1, vec2)

test_corpus[1774]
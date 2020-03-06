import gensim
import gensim.downloader as api
import json

# Download dataset
dataset = api.load("text8")
ldata = [d for d in dataset]

with open('OPTIMED_clean_preprocessed.txt', 'r', encoding = 'utf8') as file:
    data = json.load(file)

data = [list(ls.split(" ")) for ls in data]


# Create the tagged document needed for Doc2Vec
def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

train_data = list(create_tagged_document(data))
train_data2 = list(create_tagged_document(ldata))

#test_data

print(train_data[:1])

# Init the Doc2Vec model
cores = multiprocessing.cpu_count()


model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
#model.scan_vocab(train_data)

# Build the Volabulary
model.build_vocab(train_data)

# Train the Doc2Vec model
model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)

print(model.infer_vector(['australian', 'captain', 'elected', 'to', 'bowl']))

##########
ranks = []
second_ranks = []
for doc_id in range(len(train_data)):
    inferred_vector = model.infer_vector(train_data[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])

import collections

counter = collections.Counter(ranks)
print(counter)

print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_data[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_data[sims[index][0]].words)))
    
import random
doc_id = random.randint(0, len(train_data) - 1)

# Compare and print the second-most-similar document
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_data[doc_id].words)))
sim_id = second_ranks[doc_id]
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_dats[sim_id[0]].words)))

# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(test_data) - 1)
inferred_vector = model.infer_vector(test_data[doc_id])
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_data[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_data[sims[index][0]].words)))
import pandas as pd
import numpy as np
import fasttext
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
import itertools
import datetime

data = pd.read_csv('test.csv', encoding = 'utf8')


train_df = data.iloc[:301,:]
test_df = data.iloc[301:,:]
    
vocabulary = dict()
inverse_vocabulary = ['<unk>']  

#load pre-trained model 
fasttxt = fasttext.load_model('cc.cs.300.bin')

doc_cols = ['doc1', 'doc2']

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#stops = set(nltk.corpus.stopwords.words('english')) #temporary stopword si spravit!!!! ..... pomocou tf-idf? 

#transform words into words ID
# Iterate over the questions only of both training and test datasets
for dataset in [train_df, test_df]:
    for index, row in dataset.iterrows():

        # Iterate through the text of both questions of the row
        for document in doc_cols:

            d2n = []  # document numbers representation
            for word in UDPipe_preprocessing(row[document]):

                # Check for unwanted words
                if word not in fasttxt.words: #word in stops and
                    continue

                if word not in vocabulary:
                    d2n.append(len(inverse_vocabulary))
                    vocabulary[word] = len(inverse_vocabulary)
                    inverse_vocabulary.append(word)
                else:
                    d2n.append(vocabulary[word])

            # Replace questions with lists of word indices
            dataset.set_value(index, document, d2n) #__main__:22: FutureWarning: set_value is deprecated and will be removed in a future release. Please use .at[] or .iat[] accessors instead

#vocabulary: word + word_ID
#inverse_vocabulary: list of words 

embedding_dim = 300
# This will be the embedding matrix
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  
embeddings[0] = 0  # So that the padding will be ignored


# Build the embedding matrix
for word, index in vocabulary.items():
    if word in fasttxt.words:
        embeddings[index] = fasttxt.get_word_vector(word)

max_seq_length = max(train_df.doc1.map(lambda x: len(x)).max(),
                     train_df.doc2.map(lambda x: len(x)).max(),
                     test_df.doc1.map(lambda x: len(x)).max(),
                     test_df.doc2.map(lambda x: len(x)).max())

# Split to train validation
validation_size = 40
training_size = len(train_df) - validation_size

X = train_df[doc_cols]
Y = train_df['similar']

X_train, X_validation, Y_train, Y_validation = sklearn.model_selection.train_test_split(X, Y, test_size=validation_size)

# Split to dicts22
X_train = {'left': X_train.doc1, 'right': X_train.doc2}
X_validation = {'left': X_validation.doc1, 'right': X_validation.doc2}
X_test = {'left': test_df.doc1, 'right': test_df.doc2}

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values


#model setting parameters 
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 25

# Zero padding
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
   dataset[side] = tf.keras.preprocessing.sequence.pad_sequences(dataset[side],  maxlen= max_seq_length)        

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

def exponent_neg_manhattan_distance(left, right):
    return tf.keras.backend.exp(-tf.keras.backend.sum(tf.keras.backend.abs(left-right), axis=1, keepdims=True))    
    
# The visible layer
left_input = tf.keras.Input(shape=(max_seq_length,), dtype='int32')
right_input = tf.keras.Input(shape=(max_seq_length,), dtype='int32')

embedding_layer = tf.keras.layers.Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length= max_seq_length, trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = tf.keras.layers.LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = tf.keras.layers.Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Pack it all up into a model
malstm = tf.keras.models.Model([left_input, right_input], [malstm_distance])  

# Adadelta optimizer, with gradient clipping by norm
optimizer = tf.keras.optimizers.Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

training_start_time = time()

#input - sequence of words ID 
malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size = batch_size, epochs=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right']], Y_validation))

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

# Plot accuracy
plt.plot(malstm_trained.history['accuracy'])
plt.plot(malstm_trained.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()



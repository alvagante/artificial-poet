# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from keras.layers import *
from keras.utils import to_categorical

import datetime
rundate = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
import sys
author='carducci'
text_file_path = 'texts/' + author + '-mini.txt'
titolo=sys.argv[1] if len(sys.argv) > 1 else "notte"
lenght=sys.argv[2] if len(sys.argv) > 2 else "30"
epochs=sys.argv[3] if len(sys.argv) > 3 else "20"

summary = "Start datetime: " + rundate + " | Author: " + author + " | Title: " + titolo + " | Epochs: " + epochs
print(summary)


def get_raw_data_from_file( path ):
    text = str()
    with open(path, "r") as fd:
        text += fd.read()
    return text

raw_text = get_raw_data_from_file(text_file_path)

tokenizer = Tokenizer()

corpus = raw_text.split( "\n\n" )
tokenizer.fit_on_texts(corpus)
total_words = len( tokenizer.word_index ) + 1

input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)
    
sequence_lengths = list()
for x in input_sequences:
    sequence_lengths.append( len( x ) )
max_sequence_len = max( sequence_lengths )

input_sequences = np.array(pad_sequences(input_sequences,
                                         maxlen=max_sequence_len+1, padding='pre'))
x, y = input_sequences[:, :-1], input_sequences[:, -1]
y = keras.utils.to_categorical(y, num_classes=total_words)

dropout_rate = 0.3
activation_func = keras.activations.relu

SCHEMA = [

    Embedding( total_words , 10, input_length=max_sequence_len ),
    LSTM( 32 ) ,
    Dropout(dropout_rate),
    Dense( 32 , activation=activation_func ) ,
    Dropout(dropout_rate),
    Dense( total_words, activation=tf.nn.softmax )

]

model = keras.Sequential(SCHEMA)
model.compile(
    optimizer=keras.optimizers.Adam() ,
    loss=keras.losses.categorical_crossentropy ,
    metrics=[ 'accuracy' ]
)
model.summary()
model.fit(
    x,
    y,
    batch_size=50 ,
    epochs=int(epochs),
    verbose=2,
)

def predict(seed_text , seed=10 ):

    for i in range( seed ):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=
        max_sequence_len , padding='pre')
        predicted = model.predict_classes(token_list, verbose=0 )
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word

    return seed_text

prediction = predict( titolo, int( lenght ) ) 

with open('output.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(prediction)

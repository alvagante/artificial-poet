# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 21:41:14 2020

@author: alvag
"""


#Uninstall
#!pip uninstall tensorflow
# Install version 1.0.0
#!pip install tensorflow==1.14


import numpy as np
import tensorflow as tf

# Run this cell to mount your Google Drive.
#from google.colab import drive
#drive.mount('/content/drive')

#!pip install keras


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow import keras
#import tensorflow.contrib.keras as keras
from keras.layers import *
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical

text_file_path = 'neruda_poem.txt'


def get_raw_data_from_file( path ):
    text = str()
    with open(path, "r") as fd:
        text += fd.read()
    return text

raw_text = get_raw_data_from_file( text_file_path)

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
"""
model.fit(
    x,
    y,
    batch_size=50 ,
    epochs=150,
    verbose=2,
)
"""

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

print( 
  predict( 
    input( 'Enter some starter text ( I want ... ) : ') , 
    int( input( 'Enter the desired length of the generated sentence : '))  
  ) 
)




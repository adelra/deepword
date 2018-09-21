from __future__ import print_function

import argparse
import re
from pathlib import Path

import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Bidirectional, Concatenate, Dense, Embedding
from keras.models import Model

parser = argparse.ArgumentParser(description='Deepword process.')
parser.add_argument('--batch_size', type=int,
                    help='Batch Size', default=64)
parser.add_argument('--epochs', type=int, help='Number of Epochs to train on', default=20)
parser.add_argument('--samples', type=int, help='Number of training samples to train on', default=42068)
parser.add_argument('--data', type=str, help='Data path', default='data.txt')
parser.add_argument('--latent_dim', type=int, help='Size of the Latent Dimensionality in the encoding space',
                    default=256)
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default='checkpoint.hdf5')

args = parser.parse_args()

batch_size = args.__dict__["batch_size"]
epochs = args.__dict__['epochs']  # Number of epochs to train for.
latent_dim = args.__dict__['latent_dim']  # Latent dimensionality of the encoding space.
num_samples = args.__dict__['samples']  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = args.__dict__['data']
checkpoint = args.__dict__['checkpoint']
# Vectorize the data.

with open(data_path, 'r', encoding='utf-8') as _temp_file:
    _temp_file_lines = _temp_file.readlines()
    max_sent_length = max(_temp_file_lines, key=len).split(" ").__len__()
    _temp_file.close()

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
    all_words = set()
    input_words = []
    target_words = []

    input_token_index = {}
    target_token_index = {}

    encoder_input_data = np.zeros((len(lines), max_sent_length + 1), dtype='float32')
    decoder_input_data = np.zeros((len(lines), max_sent_length + 1), dtype='float32')
    # TODO: add third dimension

for line in lines:
    for word in line.split(" "):
        if word not in all_words:
            all_words.add(word)
print(len(lines), max_sent_length + 1, len(all_words))
decoder_target_data = np.zeros((len(lines), max_sent_length + 1, len(all_words)), dtype='float32')
for line in lines:
    line = re.sub(r"\n|\r\n|\t|[ ]+|", "", line)
    words = line.split(' ')
    _words = line.split(' ')
    for index, word in enumerate(words):
        _words.pop(_words.index(word))
        # texts_dic[word] = words
        # encoder_input_data[i, t] = input_token_index[word]
        # TODO: length e input bayad be andaze len(words) bashe va output 1 doone
        # print(encoder_input_data.shape, encoder_input_data)
        # encoder_input_data = np.insert(encoder_input_data, 0, [1, 2], axis=0)  # _words index
        # encoder_input_data = np.insert(encoder_input_data, 1, [3, 4], axis=1)  # _words index
        for target_index, target in enumerate(_words):
            input_words.append(word)
            target_words.append(target)
            encoder_input_data[index, target_index] = 1
            # TODO: target text inja hamun kalamamoone, yani be tartib index haye target ro behesh midim
            input_token_index[index] = word
            target_token_index[target] = target_index
            decoder_input_data[index, target_index] = 1
            decoder_target_data[index, target_index - 1, target_token_index[word]] = 1

encoder_inputs = Input(shape=(None,), name="encoder_inputt")
x = Embedding(max_sent_length + 1, latent_dim)(encoder_inputs)
encoder = Bidirectional(LSTM(latent_dim, return_state=True), name="Encoder")
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(x)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,), name="decoder_inputs")
y = Embedding(max_sent_length + 1, latent_dim)(decoder_inputs)

decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True, name="decod")
decoder_outputs, _, _ = decoder_lstm(y, initial_state=encoder_states)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
decoder_dense = Dense(len(all_words), activation='softmax', name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
checkpointer = ModelCheckpoint(filepath=checkpoint, verbose=1, period=1)
optmzr = optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer=optmzr,
              loss='MSE',
              metrics=['accuracy'])  # categorical_crossentropy sparse_categorical_crossentropy opt=rmsprop
model.summary()
checkpoint_path = Path(checkpoint)
if checkpoint_path.is_file():
    model.load_weights(checkpoint)
    print("Previous checkpoint found, loading weights.")
else:
    # TODO: add accuracy
    # model.fit([data_gen()[0], data_gen()[1]], data_gen()[2], batch_size=batch_size, epochs=epochs, validation_split=0.2,
    #           callbacks=[checkpointer])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2, callbacks=[checkpointer])
# Save model
model.summary()
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states
# TODO: 8 sob esfehan, 7/7
# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(latent_dim * 2,))
decoder_state_input_c = Input(shape=(latent_dim * 2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs, name="decoder")
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
decoder_model.summary()
# Reverse-lookup token index to decode sequences back to
# something readable.


reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


# TODO: fix decode
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, all_words))
    # Populate the first character of target sequence with the start character.
    # target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        # if (sampled_char == '\n' or
        #         len(decoded_sentence) >= max_decoder_seq_length):
        #     stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_words[seq_index])
    print('Decoded sentence:', decoded_sentence)

from __future__ import print_function

import argparse
from pathlib import Path

import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Bidirectional, Concatenate, Dense, Embedding
from keras.models import Model
from keras.utils import plot_model

import iterator
import reader

parser = argparse.ArgumentParser(description='Deepword process.')
parser.add_argument('--batch_size', type=int,
                    help='Batch Size', default=512)
parser.add_argument('--epochs', type=int, help='Number of Epochs to train on', default=20)
parser.add_argument('--data', type=str, help='Data path', default='data.txt')
parser.add_argument('--latent_dim', type=int, help='Size of the Latent Dimensionality in the encoding space',
                    default=256)
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default='checkpoint.hdf5')

args = parser.parse_args()

batch_size = args.__dict__["batch_size"]
epochs = args.__dict__['epochs']  # Number of epochs to train for.
latent_dim = args.__dict__['latent_dim']  # Latent dimensionality of the encoding space.
# Path to the data txt file on disk.
data_path = args.__dict__['data']
checkpoint = args.__dict__['checkpoint']
# Vectorize the data.

sequence = reader.read(data_path)
word_occurrence = reader.word_dict(sequence)
word_occurrence = reader.prune_occurrence(word_occurrence, 1)
word2index, index2word = reader.word_index(word_occurrence)
sequence = reader.re_index(sequence, word2index)
sequence = np.asarray(sequence, dtype=np.int)
vocab_size = len(word_occurrence)
window_size = 50
negative_samples = 1
seed = 1

batch_iterator = iterator.batch_iterator(sequence, window_size, negative_samples, batch_size, seed)


encoder_inputs = Input(shape=(1,), name="encoder_inputt")
x = Embedding(vocab_size, latent_dim, name="embedding_encoder")(encoder_inputs)
encoder = Bidirectional(LSTM(latent_dim, return_state=True), name="Encoder")
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(x)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(1,), name="decoder_inputs")
y = Embedding(vocab_size, latent_dim, name="embedding_decoder")(decoder_inputs)

decoder_lstm = LSTM(latent_dim * 2, return_sequences=False, return_state=True, name="decod")
decoder_outputs, _, _ = decoder_lstm(y, initial_state=encoder_states)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
decoder_dense = Dense(1, activation='softmax', name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
checkpointer = ModelCheckpoint(filepath=checkpoint, verbose=1, period=1)
model.compile(optimizer="SGD",
              loss='MSE',
              metrics=['accuracy'])  # categorical_crossentropy sparse_categorical_crossentropy opt=rmsprop
model.summary()

plot_model(model, "model.png", show_shapes=True, show_layer_names=True)
checkpoint_path = Path(checkpoint)
class_weight = {1: 1.0, 0: 1.0}
sequence_length = len(sequence)
steps_per_epoch = (sequence_length * (
                window_size * 2.0) + sequence_length * negative_samples) / batch_size
if checkpoint_path.is_file():
    model.load_weights(checkpoint)
    print("Previous checkpoint found, loading weights.")
else:
    model.fit_generator(batch_iterator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        class_weight=class_weight,
                        max_queue_size=100,
                        callbacks=[checkpointer])
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
# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(latent_dim * 2,))
decoder_state_input_c = Input(shape=(latent_dim * 2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
x_decoder = Embedding(vocab_size, latent_dim, name="embedding_encoder")(decoder_inputs)

decoder_outputs, state_h, state_c = decoder_lstm(
    x_decoder, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
decoder_model.summary()
# Reverse-lookup token index to decode sequences back to
# something readable.




# TODO: fix decode
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,))
    # Populate the first character of target sequence with the start character.
    # target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    decoded_sentence = ''
    output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

    sampled_token_index = np.argmax(output_tokens[0, -1])
    sampled_char = index2word[sampled_token_index]
    decoded_sentence += sampled_char

    target_seq = np.zeros((1, 1, 1))
    target_seq[0, 0, sampled_token_index] = 1.

    states_value = [h, c]

    return decoded_sentence


for seq_index in range(10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq_generate = next(batch_iterator)
    input_seq = input_seq_generate[0][0]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', index2word[seq_index])
    print('Decoded sentence:', decoded_sentence)

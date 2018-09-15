from __future__ import print_function

import argparse
from pathlib import Path

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Bidirectional, Concatenate, Dense, Multiply
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
input_texts = []
target_texts = []
max_encoder_seq_length = 0
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        if len(line) > max_encoder_seq_length:
            max_encoder_seq_length = len(line)
        else:
            pass
        words = line.split(' ')
        input_text_list = []
        target_text_list = []
        for index, word in enumerate(words):
            num_indexes = []
            for item in range(len(words)):
                num_indexes.append(item)
            num_indexes.pop(words.index(word))
            target_texts.append(word)
            target_text_list.append(word)
            for item in num_indexes:
                input_texts.append(words[item])
                input_text_list.append(words[item])

num_encoder_tokens = len(input_texts)
num_decoder_tokens = len(target_texts)

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)

input_token_index = dict(
    [(word, i) for i, word in enumerate(input_texts)])
target_token_index = dict(
    [(word, i) for i, word in enumerate(target_texts)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    # for t, char in enumerate(input_text):
    encoder_input_data[i, input_token_index[input_text]] = 1.
    for t in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[target_text]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
encoder_inputs = Input(shape=(None, max_encoder_seq_length))

# attention
attention_probs = Dense(num_encoder_tokens, activation='softmax', name='attention_probs')(encoder_inputs)
# attention_mul = Multiply([encoder_inputs, attention_probs], output_shape=(None, num_encoder_tokens),
#                          name='attention_mul')
# attn = Multiply()[encoder_inputs, attention_probs]
attention_mul = Multiply()([encoder_inputs, attention_probs])

encoder = Bidirectional(LSTM(latent_dim, return_state=True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(attention_mul)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# Callback
checkpointer = ModelCheckpoint(filepath=checkpoint, verbose=1, period=1)

# Run training
model.compile(optimizer='adagrad',
              loss='categorical_crossentropy')  # categorical_crossentropy sparse_categorical_crossentropy opt=rmsprop
checkpoint_path = Path(checkpoint)

if checkpoint_path.is_file():
    model.load_weights(checkpoint)
    print("Previous checkpoint found, loading weights.")
else:
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

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim * 2,))
decoder_state_input_c = Input(shape=(latent_dim * 2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
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


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
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
        if (sampled_char == '\n' or
                len(decoded_sentence) >= max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
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
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)

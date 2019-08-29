import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda, Bidirectional
from keras.engine import Input, Model, InputSpec
from keras.layers import Dense, Dropout, Conv1D
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Concatenate, LeakyReLU, concatenate, MaxPool1D, GlobalMaxPool1D, add
from keras.layers import Dense, Embedding, Input, Masking, Dropout, MaxPooling1D, Lambda, BatchNormalization
from keras.layers import LSTM, TimeDistributed, AveragePooling1D, Flatten, Activation, ZeroPadding1D, UpSampling1D
from keras.optimizers import Adam, rmsprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.layers import Conv1D, GlobalMaxPooling1D, ConvLSTM2D, Bidirectional, RepeatVector
from keras import regularizers
from keras.utils import plot_model, to_categorical
from keras.preprocessing.text import Tokenizer

from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import model_from_json
import os
import pydot
import graphviz


# -------------------------- set gpu using tf ---------------------------
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# -------------------  start importing keras module ---------------------
import keras.backend.tensorflow_backend as K


EPCOHS = 100  # an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 500  # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.
INPUT_DIM = 4  # a vocabulary of 4 words in case of fnn sequence (ATCG)
OUTPUT_DIM = 50  # Embedding output
RNN_HIDDEN_DIM = 62
DROPOUT_RATIO = 0.2  # proportion of neurones not used for training
MAXLEN = 250  # cuts text after number of these characters in pad_sequences
checkpoint_dir = 'checkpoints/cnn'
os.path.exists(checkpoint_dir)

from helpers.data_helpers import *

def letter_to_index(letter):
    _alphabet = 'ATGC'
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), None)


# def load_data(input_file, test_split=0.1, maxlen=MAXLEN):
#     print('Loading data...')
#     df = pd.read_csv(input_file)
#     # df = df.drop(columns=['id', 'seq', 'len', 'class', 'seq_cutted', 'kmers'])
#     print(df.columns)
#     df.columns = ['sequence', 'target']
#
#     df['sequence'] = df['sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
#     df = df.reindex(np.random.permutation(df.index))
#
#     train_size = int(len(df) * (1 - test_split))
#     X_train = df['sequence'].values[:train_size]
#     y_train = np.array(df['target'].values[:train_size])
#     X_test = np.array(df['sequence'].values[train_size:])
#     y_test = np.array(df['target'].values[train_size:])
#
#     print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
#     print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))
#     return pad_sequences(X_train, maxlen=maxlen), y_train, pad_sequences(X_test, maxlen=maxlen), y_test
#

def create_lstm_rna_seq(input_length, rnn_hidden_dim=RNN_HIDDEN_DIM, output_dim=OUTPUT_DIM, input_dim=INPUT_DIM,
                        dropout=DROPOUT_RATIO):
    model = Sequential()
    model.add(Embedding(input_dim=INPUT_DIM, output_dim=output_dim, input_length=input_length, name='embedding_layer'))
    model.add(Bidirectional(LSTM(rnn_hidden_dim, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(rnn_hidden_dim)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model


def create_cnn_rna_seq(sequence_length, train_vocabulary,
                               embedding_dim, filter_sizes, num_filters,
                               drop, epochs, batch_size):
    # this returns a tensor
    print("Creating Model...")
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

    conv_0 = Conv1D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv1D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv1D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool1D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool1D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool1D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=2, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    return model


def create_plots(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_stats/cnn_accuracy.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_stats/cnn_loss.png')
    plt.clf()


if __name__ == '__main__':
    input_file = './processed_data/kmer_map.csv'
    # train



    # X_train, y_train, X_test, y_test = load_data(input_file)
    arr_data = load_data()

    X_train, y_train, train_vocabulary, train_vocabulary_inv = arr_data[0]
    X_test, y_test, test_vocabulary, test_vocabulary_inv = arr_data[1]

    sequence_length = X_train.shape[1]  # 56
    vocabulary_size = len(train_vocabulary)  # 18765
    embedding_dim = 256
    filter_sizes = [3, 4, 5]
    num_filters = 512
    drop = 0.5
    epochs = 100
    batch_size = 16


    model = create_cnn_rna_seq(sequence_length, train_vocabulary,
                               embedding_dim, filter_sizes, num_filters,
                               drop, epochs, batch_size)


    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])



    # save checkpoint
    filepath= checkpoint_dir + "/cnn_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print ('Fitting model...')

    history = model.fit(X_train, y_train, batch_size=batch_size,
        epochs=EPCOHS, callbacks=callbacks_list, validation_split = 0.1, verbose = 1)
                        # , validation_data=(X_test, y_test))

    # serialize model to JSON
    model_json = model.to_json()
    with open("models_arch/cnn_model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("models_weights/cnn_model_weights.h5")
    print("Saved model to disk")
    create_plots(history)
    plot_model(model, to_file='cnn_model.png')

    # validate model on unseen data
    score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print('Validation score:', score)
    print('Validation accuracy:', acc)

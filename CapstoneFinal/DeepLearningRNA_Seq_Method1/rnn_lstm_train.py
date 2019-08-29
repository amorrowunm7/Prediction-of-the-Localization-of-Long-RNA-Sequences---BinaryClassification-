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
from keras.layers import Concatenate, LeakyReLU, concatenate, MaxPool1D,GlobalMaxPool1D,add
from keras.layers import Dense, Embedding, Input, Masking, Dropout, MaxPooling1D,Lambda, BatchNormalization
from keras.layers import LSTM, TimeDistributed, AveragePooling1D, Flatten,Activation,ZeroPadding1D, UpSampling1D
from keras.optimizers import Adam, rmsprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint, CSVLogger
from keras.layers import Conv1D, GlobalMaxPooling1D, ConvLSTM2D, Bidirectional,RepeatVector
from keras import regularizers
from keras.utils import plot_model, to_categorical
from keras.preprocessing.text import Tokenizer


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

EPCOHS = 100 #  an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 500 # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.
INPUT_DIM = 4 # a vocabulary of 4 words in case of fnn sequence (ATCG)
OUTPUT_DIM = 50 # Embedding output
RNN_HIDDEN_DIM = 62
DROPOUT_RATIO = 0.2 # proportion of neurones not used for training
MAXLEN = 250 # cuts text after number of these characters in pad_sequences
checkpoint_dir ='checkpoints/lstm'
os.path.exists(checkpoint_dir)



def letter_to_index(letter):
    _alphabet = 'ATGC'
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), None)

def load_data(input_file, test_split = 0.1, maxlen = MAXLEN):
    print ('Loading data...')
    df = pd.read_csv(input_file)
    # df = df.drop(columns=['id', 'seq', 'len', 'class', 'seq_cutted', 'kmers'])
    print(df.columns)
    df.columns = ['sequence','target']

    df['sequence'] = df['sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    df = df.reindex(np.random.permutation(df.index))

    train_size = int(len(df) * (1 - test_split))
    X_train = df['sequence'].values[:train_size]
    y_train = np.array(df['target'].values[:train_size])
    X_test = np.array(df['sequence'].values[train_size:])
    y_test = np.array(df['target'].values[train_size:])

    print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))
    return pad_sequences(X_train, maxlen=maxlen), y_train, pad_sequences(X_test, maxlen=maxlen), y_test

def create_lstm_rna_seq(input_length, rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
    model = Sequential()
    model.add(Embedding(input_dim = INPUT_DIM, output_dim = output_dim, input_length = input_length, name='embedding_layer'))
    model.add(Bidirectional(LSTM(rnn_hidden_dim, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(rnn_hidden_dim)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

def create_conv_rna_seq(input_length, rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
    model = Sequential()
    # model.add(Embedding(input_dim = INPUT_DIM, output_dim = output_dim, input_length = input_length, name='embedding_layer'))
    # output_dim
    forward_lstm = LSTM(input_dim=INPUT_DIM, output_dim=output_dim, return_sequences=True)
    backward_lstm = LSTM(input_dim=INPUT_DIM, output_dim=output_dim, return_sequences=True)
    brnn = Bidirectional(forward=forward_lstm, backward=backward_lstm, return_sequences=True)
    
    model.add(Conv1D(input_dim=INPUT_DIM,
                            input_length=input_length,
                            nb_filter=320,
                            filter_length=26,
                            border_mode="valid",
                            activation="relu",
                            subsample_length=1))

    model.add(MaxPooling1D(pool_length=13, stride=13))
    model.add(Dropout(0.2))
    model.add(brnn)
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(input_dim=75*640, output_dim=925))
    model.add(Activation('relu'))
    model.add(Dense(input_dim=925, output_dim=919))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")    
    return model



def create_plots(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_stats/lstm_accuracy.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_stats/lstm_loss.png')
    plt.clf()

if __name__ == '__main__':

    input_file = './processed_data/kmer_map.csv'
    # train
    X_train, y_train, X_test, y_test = load_data(input_file)
    model = create_lstm_rna_seq(len(X_train[0]))

    
    # save checkpoint
    filepath= checkpoint_dir + "/lstm_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print ('Fitting model...')
    class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    print(class_weight)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, class_weight=class_weight,
        epochs=EPCOHS, callbacks=callbacks_list, validation_split = 0.1, verbose = 1)

    # serialize model to JSON
    model_json = model.to_json()
    with open("models_arch/lstm_model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("models_weights/lstm_model_weights.h5")
    print("Saved model to disk")
    create_plots(history)
    plot_model(model, to_file='lstm_model.png')

    # validate model on unseen data
    score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print('Validation score:', score)
    print('Validation accuracy:', acc)

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


from helpers.prepare_data import *


EPCOHS = 100  # an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 500  # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.
INPUT_DIM = 4  # a vocabulary of 4 words in case of fnn sequence (ATCG)
OUTPUT_DIM = 50  # Embedding output
RNN_HIDDEN_DIM = 62
DROPOUT_RATIO = 0.2  # proportion of neurones not used for training
MAXLEN = 250  # cuts text after number of these characters in pad_sequences
checkpoint_dir = 'checkpoints/lstm'
os.path.exists(checkpoint_dir)


def letter_to_index(letter):
    _alphabet = 'ATGC'
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), None)


def load_data(input_file, test_split=0.1, maxlen=MAXLEN):
    print('Loading data...')
    df = pd.read_csv(input_file)
    # df = df.drop(columns=['id', 'seq', 'len', 'class', 'seq_cutted', 'kmers'])
    print(df.columns)
    df.columns = ['sequence', 'target']

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


def create_feedforward_rna_seq(X_train, max_features):
    # the input must have the number of features
    inp1 = Input(shape=(X_train.shape[1],), dtype='float')
    emb = Embedding(max_features, 120)(inp1)
    main = Dense(64)(emb)
    # Dropouts are important t prevent the overfitting
    main = Dropout(0.5)(main)
    # Batch normalization allow faster convergence
    main = BatchNormalization()(main)
    main = Dense(64)(main)
    main = Dropout(0.5)(main)
    main = BatchNormalization()(main)
    main = GlobalMaxPooling1D()(main)
    main = Dense(32)(main)
    # main = Dropout(0.5)(main)
    # main = Dense(256)(main)
    # main = Dropout(0.5)(main)
    # Simoid function is a must in binary classification
    out = Dense(1, activation='sigmoid')(main)
    model = Model(inputs=[inp1], outputs=[out])
    # Slower leraning rate is important in small dataset < 0.001
    optimizer = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model




def create_plots(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_stats/feedforward_accuracy.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_stats/feedforward_loss.png')
    plt.clf()


if __name__ == '__main__':
    input_file = './processed_data/kmer_map.csv'
    # train
    # X_train, y_train, X_test, y_test = load_data(input_file)

    k4mer_stride_df = pd.read_csv('./raw_data/NuclearCytosolLncRNAs_ALL_4mer_stride1_tokens.csv.count.csv', index_col=0)
    # k8mer_stride_df = pd.read_csv('./raw_data/NuclearCytosolLncRNAs_ALL_8mer_stride1_tokens.csv.count.csv',index_col=0)


    RNA_df, max_len, KMER_SIZE = make_k_mer_map(k4mer_stride_df, './raw_data/lncRNA_amanda.fasta')
    # print(RNA_df.head())
    train, test, y_train, y_test = make_train_test_dfs(RNA_df)
    # print(train.head())
    X_train, X_test = make_train_test_tokenizer(train, test)

    X_train, X_test = pad_dfs(X_train, X_test, max_len)

    max_features = X_train.max() + 1
    Y_train, Y_test = categorize_dfs(y_train, y_test)

    #calculate class weights
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(Y_train.argmax(-1)),
                                                      Y_train.argmax(-1))

    #
    model = create_feedforward_rna_seq(X_train, max_features)
    #
    # save checkpoint
    filepath = checkpoint_dir + "/feedforward_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print('Fitting model...')
    class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    # print(class_weight)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, class_weight=class_weight,
                        epochs=EPCOHS, callbacks=callbacks_list, validation_split=0.1, verbose=1)
    #
    # serialize model to JSON
    model_json = model.to_json()
    with open("models_arch/feedforward_model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("models_weights/feedforward_model_weights.h5")
    print("Saved model to disk")
    create_plots(history)
    plot_model(model, to_file='feedforward_model.png')

    # validate model on unseen data
    score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print('Validation score:', score)
    print('Validation accuracy:', acc)
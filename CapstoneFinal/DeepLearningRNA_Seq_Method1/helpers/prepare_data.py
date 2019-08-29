import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import Bio
from Bio import Seq, SeqIO
from Bio.Alphabet import generic_dna
import itertools
from sklearn.utils import class_weight


from keras.utils import plot_model, to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



def RNA_2_csv(path):
    #Reading  database as a panda dataframe
    reads=[]
    with open(path, 'r') as file:
        for record in SeqIO.parse(file, 'fasta'):
            id_ = record.description.split(' ')[0]
            seq_ = str(record.seq)
            reads.append([id_,seq_,len(record.seq)])
    df = pd.DataFrame(reads,columns=['id','seq','len'])
    return df


# Convert a sequence of charaters to a sequence of kmers
def seq_to_kmer(seq, KMER_SIZE):
    return ' '.join([seq[i:i+KMER_SIZE] for i in range(len(seq)-KMER_SIZE)]).lower()

def make_kmer_map_to_csv(k_mer_stride_df, fast_path, KMER_SIZE=6):
    print("running make_kmer_map_to_csv")
    RNA_df = RNA_2_csv(fast_path)
    ## Map the classes from df to RNA_df
    RNA_df.index = RNA_df['id']
    k_mer_stride_df['id'] = k_mer_stride_df.index
    RNA_df['class'] = RNA_df['id'].map(k_mer_stride_df.set_index('id')['class'])
    # Drop any rows with NAin the class
    RNA_df = RNA_df.dropna()

    # convert the classes to 0 and 1
    RNA_df['class'] = pd.factorize(RNA_df['class'])[0]
    max_len = int(RNA_df.len.std()*2)
    RNA_df['seq_cutted'] = RNA_df['seq'].apply(lambda x: x[0:max_len])
    RNA_df['kmers'] = RNA_df['seq_cutted'].apply(lambda x: seq_to_kmer(x, KMER_SIZE))

    RNA_df=RNA_df.drop(columns= ['id', 'len', 'seq_cutted', 'kmers'])

    RNA_df.to_csv('./processed_data/kmer_map.csv', encoding='utf-8', index=False)
    print("saved kmer_map csv")
    return RNA_df, max_len, KMER_SIZE


def make_k_mer_map(k_mer_stride_df, fast_path, KMER_SIZE=6):

    RNA_df = RNA_2_csv(fast_path)
    ## Map the classes from df to RNA_df
    RNA_df.index = RNA_df['id']
    k_mer_stride_df['id'] = k_mer_stride_df.index
    RNA_df['class'] = RNA_df['id'].map(k_mer_stride_df.set_index('id')['class'])

    # Drop any rows with NAin the class
    RNA_df = RNA_df.dropna()

    # convert the classes to 0 and 1
    RNA_df['class'] = pd.factorize(RNA_df['class'])[0]

    max_len = int(RNA_df.len.std()*2)

    RNA_df['seq_cutted'] = RNA_df['seq'].apply(lambda x: x[0:max_len])
    RNA_df['kmers'] = RNA_df['seq_cutted'].apply(lambda x: seq_to_kmer(x, KMER_SIZE))

    return RNA_df, max_len, KMER_SIZE


def make_train_test_dfs(RNA_df):

    ## Validation data is important for neural network backpropagation
    X=RNA_df.drop(columns=['class','id','len','seq','seq_cutted'])
    y=RNA_df.loc[:,'class']
    train, test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=15)

    return train, test, y_train, y_test


def make_train_test_tokenizer(train, test):

    # Tokenizing
    tokenizer = Tokenizer(split=' ')
    tokenizer.fit_on_texts(train.values.tolist())
    X_train = tokenizer.texts_to_sequences(train.values.tolist())
    X_test = tokenizer.texts_to_sequences(test.values.tolist())

    return X_train, X_test


def pad_dfs(X_train, X_test, max_len):
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    return X_train, X_test


def categorize_dfs(y_train, y_test):
    # Converting the output to categorical classes
    Y_train = to_categorical(y_train)
    Y_test = to_categorical(y_test)

    return Y_train, Y_test



# k4mer_stride_df = pd.read_csv('./raw_data/NuclearCytosolLncRNAs_ALL_4mer_stride1_tokens.csv.count.csv',index_col=0)
# k8mer_stride_df = pd.read_csv('./raw_data/NuclearCytosolLncRNAs_ALL_8mer_stride1_tokens.csv.count.csv',index_col=0)


# make_kmer_map_to_csv(k4mer_stride_df, './raw_data/lncRNA_amanda.fasta')


# RNA_df, max_len, KMER_SIZE = make_k_mer_map(k4mer_stride_df, './raw_data/lncRNA_amanda.fasta')
# print(RNA_df.head())
# train, test, y_train, y_test = make_train_test_dfs(RNA_df)
# print(train)
# X_train, X_test = make_train_test_tokenizer(train, test)
# X_train, X_test = pad_dfs(X_train, X_test, max_len)

# max_features = X_train.max() + 1
# Y_train, Y_test = categorize_dfs(y_train, y_test)
#
# #calculate class weights
# class_weights = class_weight.compute_class_weight('balanced',
#                                                   np.unique(Y_train.argmax(-1)),
#                                                   Y_train.argmax(-1))
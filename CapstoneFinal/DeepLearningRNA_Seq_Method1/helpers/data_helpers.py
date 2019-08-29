import numpy as np
import re
import itertools
from collections import Counter



import gzip
import math
import os.path
from subprocess import Popen, PIPE, STDOUT

import numpy as np
import pandas as pd


def get_fastas_from_file(fasta_path, as_dict=False,
                         uppercase=False, stop_at=None):
    fastas = []
    seq = None
    header = None
    for r in (gzip.open(fasta_path) if fasta_path.endswith(".gz") else open(fasta_path)):
        if type(r) is bytes:
            r = r.decode("utf-8")
        r = r.strip()
        if r.startswith(">"):
            if seq != None and header != None:
                fastas.append([header, seq])
                if stop_at != None and len(fastas) >= stop_at:
                    break
            seq = ""
            header = r[1:]
        else:
            if seq != None:
                seq += r.upper() if uppercase else r
            else:
                seq = r.upper() if uppercase else r
    # append last fasta read by method
    if stop_at != None and len(fastas) < stop_at:
        fastas.append([header, seq])
    elif stop_at == None:
        fastas.append([header, seq])
    if as_dict:
        return {h: s for h, s in fastas}

    return pd.DataFrame({'location': [e[0] for e in fastas], 'sequence': [e[1] for e in fastas]})


def get_shape_fastas_from_file(fasta_path, as_dict=False,
                         uppercase=False, stop_at=None):
    fastas = []
    seq = None
    header = None
    for r in (gzip.open(fasta_path) if fasta_path.endswith(".gz") else open(fasta_path)):
        if type(r) is bytes:
            r = r.decode("utf-8")
        r = r.strip()
        if r.startswith(">"):
            if seq != None and header != None:
                fastas.append([header, seq])
                if stop_at != None and len(fastas) >= stop_at:
                    break
            seq = None
            header = r[1:]
        else:
            if seq != None:
                seq += "," + (r.upper() if uppercase else r)
            else:
                seq = r.upper() if uppercase else r
    # append last fasta read by method
    if stop_at != None and len(fastas) < stop_at:
        fastas.append([header, seq])
    elif stop_at == None:
        fastas.append([header, seq])
    if as_dict:
        return {h: s for h, s in fastas}

    return pd.DataFrame({'location': [e[0] for e in fastas], 'sequence': [e[1] for e in fastas]})


def get_padded_sequences(fasta_file):
    fasta = get_fastas_from_file(fasta_file)
    max_length = max([len(x) for x in fasta.sequence])
    padded_sequences = []
    for seq in fasta.sequence:
        diff = max_length - len(seq)
        n_seq = (math.floor(diff/2) * 'N') + seq + (math.ceil(diff/2) * 'N')
        padded_sequences.append(n_seq)
    fasta.sequence = padded_sequences
    return fasta


def convert_bed_to_fasta_hg19(bed_path, fasta_path, reference_genome_path, use_peak_max=False,
                              bp_flanking=50):
    '''
    Copied from Ignacio: /g/scb/zaugg/rio/EclipseProjects/zaugglab/lib/FastaAnalyzer.py
    :param bed_path: The path to our BED file
    :param fasta_path: The output fasta that will be created
    :param use_peak_max: If True, we will extract w.r.t. to peak position
    (See https://www.biostars.org/p/102710/ for format description
    :param bp_flanking: If use_peak is True, then flanking regions will
    be calculated from this file
    :return:
    '''

    args = ["/g/software/bin/bedtools", "getfasta", "-fi", reference_genome_path,
            "-fo", fasta_path]

    # create a new coordinates file with flanking sequences
    if use_peak_max:
        df = pd.read_csv(bed_path, sep='\t', index_col=False,
                         names=['chrom', 'chromStart', 'chromEnd', 'name', 'score',
                                'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts'])
        df['startFromPeak'] = df['thickStart'] - bp_flanking
        df['endFromPeak'] = df['thickStart'] + bp_flanking
        df = df[['chrom', 'startFromPeak', 'endFromPeak']]
        tsv_string = df.to_csv(header=False, sep='\t', index=False)
        args = args + ['-bed', 'stdin']

        p = Popen(args, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        x = p.communicate(input=tsv_string.encode(encoding='UTF-8'))
        x = x[0].decode('UTF-8')
        if x != '':
            print("ERROR: " + x)
    else:
        os.system(" ".join(args + ['-bed', bed_path]))


def write_fasta_file(file, sequences, descr=None):
    """
    Sequences has to be a list of strings. descr can be None than a dummy line is inserted or a list of the
    same length as sequences.
    """
    with open(file, "w") as out:
        for idx, seq in enumerate(sequences):
            if descr is None:
                out.write(">Dummy_Line\n")
            else:
                out.write(">" + str(descr[idx]) + "\n")
            out.write("".join(seq) + "\n")


def save_keras_model(model, model_path, overwrite=False):
    json_string = model.to_json()
    with open(model_path + '.json', 'w+') as f:
        f.write(json_string)
    model.save_weights(model_path + '.h5', overwrite=overwrite)


def load_keras_model(path):
    from keras.models import model_from_json
    model = model_from_json(open(path + '.json').read())
    model.load_weights(path + '.h5')
    return model


def save_scoring_file(header, values, scores, labels, file):
    if len(scores) != len(labels):
        raise ValueError("The score and label length must match!")
    if len(header) != scores.shape[3] + values.shape[2]:
        raise ValueError("The value + score width and header length must match!")

    with open(file, 'w') as output:
        output.write("\t".join(["Index", "Label"] + header) + "\n")
        for line_idx in range(0, len(scores)):
            output.write("\t".join([str(line_idx), labels[line_idx]]
                                   + ["["+",".join(map(str, values[line_idx, :, c]))+"]" for c in range(0, values.shape[2])]
                                   + ["["+",".join(map(str, scores[line_idx, 0, :, c]))+"]" for c in range(0, scores.shape[3])]))
            output.write("\n")



def read_importance_file(location):
    return pd.read_csv(location, sep="\t")


def parse_importance_df(df, col_names):
    # Iterate over every entry
    parsed_cols = []
    for name in col_names:
        col = df[name].as_matrix()
        parsed_col = np.apply_along_axis(lambda e: np.array([float(x) for x in e[0][1:-1].split(",")]), 1, col.reshape(len(col),1))
        parsed_cols.append(parsed_col)
    return np.stack(parsed_cols, 2)


def write_output_file(output_file, name, PositiveData, NegativeData, Training_Script, aucs, auprcs, importance_scores):
    with open(output_file, "w") as out:
        out.write("Name:" + str(name) + "\n")
        out.write("PositiveData:" + str(PositiveData) + "\n")
        out.write("NegativeData:" + str(NegativeData) + "\n")
        out.write("Training_Script:" + str(Training_Script) + "\n")
        out.write("AUCs:" + ",".join(map(str, aucs)) + "\n")
        out.write("AUPRCs:" + ",".join(map(str, auprcs)) + "\n")
        out.write("Importance_Scores:" + str(importance_scores) + "\n")


###########################

import numpy as np


def parse_alpha_to_seq(sequence):
    output = np.arange(len(sequence))
    for i in range(0, len(sequence)):
        snippet = sequence[i]
        if snippet == 'A':
            output[i] = 0
        elif snippet == 'C':
            output[i] = 1
        elif snippet == 'T':
            output[i] = 2
        elif snippet == 'G':
            output[i] = 3
        elif snippet == 'N':
            output[i] = -1
        else:
            raise AssertionError("Cannot handle snippet: " + snippet)
    return output


def parse_binary_seq(sequence):
    output = np.arange(len(sequence) / 2)
    for i in range(0, len(sequence), 2):
        snippet = sequence[i] + sequence[i + 1]
        if snippet == '00':
            output[int(i / 2)] = 0
        elif snippet == '01':
            output[int(i / 2)] = 1
        elif snippet == '10':
            output[int(i / 2)] = 2
        elif snippet == '11':
            output[int(i / 2)] = 3
        else:
            raise AssertionError("Cannot handle snippet: " + snippet)
    return output


def parse_binary_seq_to_alpha(sequence):
    output = ""
    for i in range(0, len(sequence), 2):
        snippet = sequence[i] + sequence[i + 1]
        if snippet == '00':
            output += 'A'
        elif snippet == '01':
            output += 'C'
        elif snippet == '10':
            output += 'G'
        elif snippet == '11':
            output += 'T'
        else:
            raise AssertionError("Cannot handle snippet: " + snippet)
    return output


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        if y[i] != -1:
            Y[i, y[i]] = 1.
    return Y


def do_one_hot_encoding(sequence, seq_length, f=parse_alpha_to_seq):
    X = np.zeros((sequence.shape[0], seq_length, 4))
    for idx in range(0, len(sequence)):
        X[idx] = to_categorical(f(sequence[idx]), 4)
    return X


def do_dinucleotide_shuffling(X, size=1):
    x_shuffled = np.repeat(X, size, 0)

    for x in range(0, x_shuffled.shape[0]):
        random_index = np.arange(0, X.shape[1]/2)
        np.random.shuffle(random_index)
        for y in range(0, int(X.shape[1]/2)):
            x_shuffled[x,y*2, ] = X[x%X.shape[0],random_index[y]*2]
            x_shuffled[x,(y*2)+1, ] = X[x%X.shape[0],(random_index[y]*2)+1]

    return x_shuffled


def generate_complementary_sequence(sequence):
    comp_seq = []
    for b in sequence:
        if b == "A":
            comp_seq.append("T")
        elif b == "T":
            comp_seq.append("A")
        elif b == "C":
            comp_seq.append("G")
        elif b == "G":
            comp_seq.append("C")
        elif b == "N":
            comp_seq.append("N")
        else:
            raise ValueError("Cannot convert base {0} to complement base!".format(b))
    return ''.join(comp_seq)


#########################

# import SequenceHelper

# from prepare_data import *


#########################


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import Bio
from Bio import Seq, SeqIO
from Bio.Alphabet import generic_dna
import itertools
from sklearn.utils import class_weight



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
    print(k_mer_stride_df.head())
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

#
#
# k4mer_stride_df = pd.read_csv('./raw_data/NuclearCytosolLncRNAs_ALL_4mer_stride1_tokens.csv.count.csv',index_col=0)
# # k8mer_stride_df = pd.read_csv('./raw_data/NuclearCytosolLncRNAs_ALL_8mer_stride1_tokens.csv.count.csv',index_col=0)
#
#
# # make_kmer_map_to_csv(k4mer_stride_df, './raw_data/lncRNA_amanda.fasta')
#
#
# RNA_df, max_len, KMER_SIZE = make_k_mer_map(k4mer_stride_df, './raw_data/lncRNA_amanda.fasta')
# print(RNA_df.head())
# train, test, y_train, y_test = make_train_test_dfs(RNA_df)
# print(train)
# X_train, X_test = make_train_test_tokenizer(train, test)
# max_features = X_train.max() + 1
# X_train, X_test = pad_dfs(X_train, X_test, max_len)
# Y_train, Y_test = categorize_dfs(y_train, y_test)
#
# #calculate class weights
# class_weights = class_weight.compute_class_weight('balanced',
#                                                   np.unique(Y_train.argmax(-1)),
#                                                   Y_train.argmax(-1))

#########################


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    # new_dt = IOHelper.get_fastas_from_file("../new_dt.csv")
    # new_dt[(new_dt['class'] == 0)]
    # new_dt[(new_dt['class'] == 1)]
    # print(new_dt)
    # positive_examples = list(open("./data/rt-polarity.pos", "r", encoding='latin-1').readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples =  list(open("./data/rt-polarity.neg", "r", encoding='latin-1').readlines())
    # negative_examples = [s.strip() for s in negative_examples]
    # # Split by words
    # x_text = positive_examples + negative_examples
    # x_text = [clean_str(sent) for sent in x_text]
    # x_text = [s.split(" ") for s in x_text]
    # # Generate labels
    # positive_labels = [[0, 1] for _ in positive_examples]
    # negative_labels = [[1, 0] for _ in negative_examples]
    # y = np.concatenate([positive_labels, negative_labels], 0)
    # print(x_text[0])
    # print(y[0])

    k4mer_stride_df = pd.read_csv('./raw_data/NuclearCytosolLncRNAs_ALL_4mer_stride1_tokens.csv.count.csv',
                                  index_col=0)

    # print(k4mer_stride_df.head())
    RNA_df, max_len, KMER_SIZE = make_k_mer_map(k4mer_stride_df, './raw_data/lncRNA_amanda.fasta')
    train, test, y_train, y_test = make_train_test_dfs(RNA_df)
    # print(train.head())
    train = [s.split(" ") for s in train['kmers']]
    test = [s.split(" ") for s in test['kmers']]

    # print(y_test)

    positive_examples = y_train[y_train == 1]
    negative_examples = y_train[y_train == 0]
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y_train = np.concatenate([positive_labels, negative_labels], 0)

    positive_examples = y_test[y_test == 1]
    negative_examples = y_test[y_test == 0]
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y_test = np.concatenate([positive_labels, negative_labels], 0)

    # X_train, X_test = make_train_test_tokenizer(train, test)
    # max_features = X_train.max() + 1
    # X_train, X_test = pad_dfs(X_train, X_test, max_len)
    # Y_train, Y_test = categorize_dfs(y_train, y_test)



    return [train, test, y_train, y_test]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    # sentences, labels = load_data_and_labels()
    train, test, y_train, y_test = load_data_and_labels()

    train_padded = pad_sentences(train)
    test_padded = pad_sentences(test)

    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    train_vocabulary, train_vocabulary_inv = build_vocab(train_padded)
    test_vocabulary, test_vocabulary_inv = build_vocab(test_padded)

    # x, y = build_input_data(sentences_padded, labels, vocabulary)
    train, y_train = build_input_data(train_padded, y_train, train_vocabulary)
    test, y_test = build_input_data(test_padded, y_test, test_vocabulary)


    return [[train, y_train, train_vocabulary, train_vocabulary_inv], [test, y_test, test_vocabulary, test_vocabulary_inv]]

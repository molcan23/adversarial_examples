import numpy as np
import re
import itertools
from collections import Counter
import constants as cs
import csv
from nltk.stem.porter import PorterStemmer

from w2v import *

import sys
import tensorflow as tf
from nltk.tokenize import sent_tokenize, word_tokenize


def load_data1(dataset):
    x, y, vocabulary, vocabulary_inv_list = load_data_for_w2v(dataset)
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}

    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * 0.9)
    X_train = x[:train_len]
    Y_train = y[:train_len]
    X_validate = x[train_len:]
    Y_validate = y[train_len:]

    return X_train, Y_train, X_validate, Y_validate, vocabulary_inv


def load_data(dataset):
    # Data Preparation
    print("Load data...")
    X_train, Y_train, X_validate, Y_validate, vocabulary_inv = load_data1(dataset)

    print("X_train shape:", X_train.shape)
    print("X_validate shape:", X_validate.shape)
    print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

    # Prepare embedding layer weights and convert inputs for static model
    embedding = train_word2vec(np.vstack((X_train, X_validate)), vocabulary_inv, num_features=cs.EMBEDDING_DIM,
                                       min_word_count=cs.MIN_WORD_COUNT, context=cs.CONTEXT)
    # embedding_weights
    X_train = np.stack([np.stack([embedding['weights'][word] for word in sentence]) for sentence in X_train])
    X_validate = np.stack([np.stack([embedding['weights'][word] for word in sentence]) for sentence in X_validate])
    print("X_train static shape:", X_train.shape)
    print("X_validate static shape:", X_validate.shape)
    return X_train, X_validate, Y_train, Y_validate, embedding


def clean_str(string):
    """
    Tokenization/string cleaning.
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


def load_data_and_labels(dataset, for_bayes=False):
    """
    Loads 'dataset' data from file, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # fixme spravne vratit validate a train
    name_of_file = cs.DATASET_PATHS[dataset]
    X_train, Y_train = [], []
    counter = 0
    if dataset == 'fake':
        with open(name_of_file, newline='') as csv_file:
            news_reader = csv.reader(csv_file)
            _ = next(csv_file)
            for row in news_reader:
                if counter >= cs.DATASET_MAX[dataset]: break
                X_train.append(row[3])
                Y_train.append([0, 1] if int(row[4]) == 1 else [1, 0])
                counter += 1
    if dataset == 'yelp':
        with open(name_of_file, newline='') as csv_file:
            news_reader = csv.reader(csv_file)
            for row in news_reader:
                if counter >= cs.DATASET_MAX[dataset]: break
                X_train.append(row[1])
                # print(row[0], type(row[0]))
                # print(row[1])
                Y_train.append([0, 1] if row[0] == "2" else [1, 0])
                # print()
                counter += 1

    X_train = [clean_str(sent) for sent in X_train]
    X_train = [s.split(" ") for s in X_train]

    cleaned_X_train = []
    porter = PorterStemmer()

    for x in X_train:
        # print(x)

        article = []
        for i in x:
            stripped = i.translate(cs.TABLE)
            if not stripped.isalpha() or stripped in cs.STOP_WORDS:
                continue
            stemmed = porter.stem(stripped)
            article.append(stemmed)

        cleaned_X_train.append(article)

    if for_bayes:
        Y_train = [0 if i[0] == 1 else 1 for i in Y_train]

    return np.array(cleaned_X_train), np.array(Y_train)


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
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data_for_w2v(dataset):
    """
    Loads and preprocessed data. Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    # sentences, labels = load_data_and_labels(dataset)
    # sentences_padded = pad_sentences(sentences)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    # x, y = build_input_data(sentences_padded, labels, vocabulary)
    # return [x, y, vocabulary, vocabulary_inv]

    sentences, labels = load_data_and_labels(dataset)
    vocabulary, vocabulary_inv = build_vocab(sentences)
    x, y = build_input_data(sentences, labels, vocabulary)
    return [tf.keras.preprocessing.sequence.pad_sequences(x, dtype='object'), y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# bag of words

csv.field_size_limit(sys.maxsize)


def load_bayes_train_data(dataset):
    """
    Loads 'dataset' data from file, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    name_of_file = cs.DATASET_PATHS[dataset]
    X_train, Y_train = [], []
    counter = 0

    if dataset == 'fake':
        with open(name_of_file, newline='') as csv_file:
            news_reader = csv.reader(csv_file)
            _ = next(csv_file)
            for row in news_reader:
                if counter >= cs.DATASET_MAX[dataset]: break
                X_train.append(row[3])
                Y_train.append(1 if int(row[4]) == 1 else 0)
                counter += 1
    if dataset == 'yelp':
        with open(name_of_file, newline='') as csv_file:
            news_reader = csv.reader(csv_file)
            for row in news_reader:
                if counter >= cs.DATASET_MAX[dataset]: break
                X_train.append(row[1])
                Y_train.append(1 if row[0] == "2" else 0)
                counter += 1

    return X_train, Y_train


def load_news_test_data(name_of_file, name_of_labels):
    """
    Same as above but for test set.
    """
    X_test, Y_test = [], []
    with open(name_of_file, newline='') as csvfile:
        news_reader = csv.reader(csvfile)
        _ = next(csvfile)
        for row in news_reader:
            X_test.append(row[3])

    with open(name_of_labels, newline='') as csvfile:
        news_reader = csv.reader(csvfile)
        _ = next(csvfile)
        for row in news_reader:
            Y_test.append(int(row[1]))

    return X_test, Y_test


def clean_text(original_text, dataset):
    """
    Removes everything unneeded. (stemming, stripping..)
    """
    data = []
    all_data = []
    # if not path.exists('stammed_' + dataset + '.txt'):
    max_art = 0

    for x in original_text:
        article = []
        for i in sent_tokenize(x):
            temp = [j.lower() for j in word_tokenize(i)]
            stripped = [w.translate(cs.TABLE) for w in temp]
            words = [word for word in stripped if word.isalpha()]
            words = [w for w in words if w not in cs.STOP_WORDS]
            porter = PorterStemmer()
            stemmed = [porter.stem(word) for word in words]
            article += stemmed
            all_data.append(stemmed)
            # max_word = len(stemmed) if len(stemmed) > max_word else max_word

        data.append(article)
        max_art = len(article) if len(article) > max_art else max_art

    np.savetxt('stammed_' + dataset + '.txt', data, fmt='%s', delimiter=",")
    # print(np.array(data).shape)
    # print(data)
    # else:
    #     # data = np.loadtxt('stammed_' + dataset + '.txt', dtype='str')
    #     # data = np.loadtxt('stammed_' + dataset + '.txt', dtype=np.object)
    #     # data = data.tolist()
    #     with open('stammed_' + dataset + '.txt', 'r') as f:
    #         whole = f.read().splitlines()
    #
    #         for article in whole:
    #             print(article)
    #             article = article[2:-2].split('], [')
    #             _article = []
    #             for sentence in article:
    #                 print(sentence)
    #                 _sentence = []
    #                 for word in sentence:
    #                     _sentence.append(word)
    #                 _article.append(_sentence)
    #                 all_data.append(_sentence)
    #             data = [_article]
    #        # data = np.genfromtxt('stammed_' + dataset + '.txt', dtype='str', delimiter=",")

    # train_size = int(len(original_text) * cs.TRAINING_PORTION)
    # print(max_word, max_art)

    return data, all_data, max_art  # , max_word


def bag_of_words(data):
    """
    For each article in data create bag representation.
    """
    bag = {}
    for article in data:
        _article = []
        for word in article:
            if word in bag:
                bag[word] += 1
            else:
                bag[word] = 1

    bag_representation = []

    for article in data:
        bag_article = []
        for word in bag:
            if word in article:
                bag_article.append(1)
            else:
                bag_article.append(0)
        bag_representation.append(bag_article)

    return bag, np.array(bag_representation)


def convert_to_bag_of_words_format(original_text, dataset):
    """
    Cleans text and converts it to bag of words format.
    """
    data, all_data, max_art = clean_text(original_text, dataset)

    bag, data = bag_of_words(data)

    train_size = int(len(original_text) * cs.TRAINING_PORTION)

    return np.array(data[:train_size]), np.array(data[train_size:cs.DATASET_MAX[dataset]]), bag

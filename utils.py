import re
import csv
import sys
import string
import itertools
from w2v import *
import tensorflow as tf
import constants as cs
import global_variables as gv
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize


def load_data1(dataset):
    x, y, vocabulary, vocabulary_inv_list = load_data_for_w2v(dataset)
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    np.random.seed()

    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * 0.9)
    X_train = x[:train_len]
    Y_train = y[:train_len]
    X_test = x[train_len:]
    Y_test = y[train_len:]

    return X_train, Y_train, X_test, Y_test, vocabulary_inv, vocabulary


def load_data(dataset):
    # Data Preparation
    print("Load data...")
    X_train, Y_train, X_test, Y_test, vocabulary_inv, vocabulary = load_data1(dataset)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

    # Prepare embedding layer weights and convert inputs for static model
    embedding = train_word2vec(np.vstack((X_train, X_test)), vocabulary_inv, num_features=cs.EMBEDDING_DIM,
                               min_word_count=cs.MIN_WORD_COUNT, context=cs.CONTEXT)
    # embedding_weights
    X_train = np.stack([np.stack([embedding['weights'][word].astype('float32') for word in sentence])
                        for sentence in X_train])
    X_test = np.stack([np.stack([embedding['weights'][word].astype('float32') for word in sentence])
                       for sentence in X_test])
    print("X_train static shape:", X_train.shape)
    print("X_test static shape:", X_test.shape)

    return X_train, X_test, Y_train, Y_test, embedding, vocabulary


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

    X_train = [clean_str(art) for art in X_train]
    data = clean_text(X_train)
    return np.array(data), np.array(Y_train)


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

    x = np.array([[vocabulary[word] for word in sentence if word in vocabulary] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data_for_w2v(dataset):
    """
    Loads and preprocessed data. Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """

    sentences, labels = load_data_and_labels(dataset)
    vocabulary, vocabulary_inv = build_vocab(sentences)
    x, y = build_input_data(sentences, labels, vocabulary)
    gv.max_article_len = max([len(i) for i in x])
    return [tf.keras.preprocessing.sequence.pad_sequences(x, padding='pre', dtype='object'), y,
            vocabulary, vocabulary_inv]


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


def clean_text(original_text):
    """
    Removes everything unneeded. (stemming, stripping..)
    """

    data = []
    all_data = []
    max_art = 0

    for x in original_text:
        article = []
        for i in sent_tokenize(x):
            temp = [j.lower() for j in word_tokenize(i)]
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in temp]
            words = [word for word in stripped if word.isalpha()]

            # stop_words = set(stopwords.words('english'))
            words = [w for w in words if not w in cs.STOP_WORDS]
            article += words
            all_data.append(words)
        data.append(article)
        max_art = len(article) if len(article) > max_art else max_art

    return data  # , all_data, max_art


def bag_of_words(data, bag={}):
    """
    For each article in data create bag representation.
    """

    if not bag:
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

    data = clean_text(original_text)

    bag, data = bag_of_words(data)

    train_size = int(len(original_text) * cs.TRAINING_PORTION)

    return np.array(data[:train_size]), np.array(data[train_size:cs.DATASET_MAX[dataset]]), bag

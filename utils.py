import sys
import csv
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec, KeyedVectors
from nltk.stem.porter import PorterStemmer
import constants as cs
from os import path

csv.field_size_limit(sys.maxsize)


def load_news_train_data(dataset):
    # fixme spravne vratit validate a train
    name_of_file = cs.DATASET_PATHS[dataset]
    X_train, Y_train = [], []
    with open(name_of_file, newline='') as csv_file:
        news_reader = csv.reader(csv_file)
        _ = next(csv_file)
        # news_reader.read()
        for row in news_reader:
            # print(row)
            X_train.append(row[3])
            Y_train.append(int(row[4]))

    return X_train[:cs.DATASET_MAX[dataset]], Y_train[:cs.DATASET_MAX[dataset]]


def load_news_test_data(name_of_file, name_of_labels):
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
            # article.append(stemmed)
            all_data.append(stemmed)
            # max_word = len(stemmed) if len(stemmed) > max_word else max_word

        data.append(article)
        max_art = len(article) if len(article) > max_art else max_art

    np.savetxt('stammed_' + dataset + '.txt', data, fmt='%s', delimiter=",")
    # print(np.array(data).shape)
    # print(data)
    # else:
    #     # fixme nacitat ostemmovane
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

    return np.array(bag_representation)


def convert_to_bag_of_words_format(original_text, dataset):
    data, all_data, max_art = clean_text(original_text, dataset)

    bag = bag_of_words(all_data)

    train_size = int(len(original_text) * cs.TRAINING_PORTION)

    return np.array(bag[:train_size]), np.array(bag[train_size:cs.DATASET_MAX[dataset]])


def con_w2v(data, model1, max_art):

    w2v_data = []

    for article in data:
        _article = []
        for word in article:
            if word in model1:
                _article.append(model1.wv[word])

        # padding
        for _ in range(max_art - len(_article)):
            _article.append(np.zeros(cs.SIZE))

        w2v_data.append(_article)

    return np.array(w2v_data)


def purify_words_not_in_model(data, model):
    pur_data = []
    max_art = 0
    for article in data:
        _article = []
        for word in article:
            if word in model:
                _article.append(word)

        max_art = len(_article) if len(_article) > max_art else max_art
        pur_data.append(_article)

    return np.array(pur_data), max_art


def convert_to_word2vec(original_text, dataset):
    data, all_data, max_art = clean_text(original_text, dataset)

    print(max_art)
    # Create CBOW model
    if not path.exists(dataset + '_model'):

        model = Word2Vec(
            all_data,
            size=cs.SIZE,
            min_count=cs.MIN_COUNT,
            window=cs.WINDOW,
            seed=1337
        )
        model.save(dataset + "_model")
        model.wv.save_word2vec_format(dataset + "_model.bin", binary=True)

    else:
        model = Word2Vec.load(dataset + "_model")
        # word_vectors = KeyedVectors.load_word2vec_format(dataset + "_model.bin", binary=True)

    data, max_art = purify_words_not_in_model(data, model)
    print(max_art)
    data = con_w2v(data, model, max_art)
    train_size = int(len(original_text) * cs.TRAINING_PORTION)

    return np.array(data[:train_size]), np.array(data[train_size:cs.DATASET_MAX[dataset]])


def save_w2v_text(file, data):
    with open(file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerows(data)


def load_w2v_text(file):
    # to read file you saved
    with open(file, 'r') as f:
        reader = csv.reader(f)
        examples = list(reader)
    orig_data = []

    for row in examples:
        new_row = []
        for word in row:
            print(word)
            word = word.replace("\n", "")[1:-1].split(" ")
            for el in word:
                print(el)
                el.replace('\n', '')
                new_row.append(float(el))
        orig_data.append(new_row)

    return examples

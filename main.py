from lstm_classifier import lstm_classifier
from naive_bayes import naive_bayes
from utils import *

dataset = 'fake'

articles, labels = load_news_train_data(dataset)

train_size = int(cs.DATASET_MAX[dataset] * cs.TRAINING_PORTION)

Y_train = np.array(labels[0: train_size])
Y_validate = np.array(labels[train_size:cs.DATASET_MAX[dataset]])

# fixme bag of words - implemented, but bugged
# naive_bayes(articles, labels, dataset)

# word2vec
X_train, X_validate = convert_to_word2vec(articles, dataset)

# LSTM classifier
lstm_classifier(X_train, X_validate, Y_train, Y_validate, dataset)

# Shallow word-level convolutional network

# Deep  character-level  convolutional  networks

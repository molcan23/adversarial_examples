from lstm_classifier import LSTM_Classifier
from shallow_CNN import ShallowCNN
from naive_bayes import naive_bayes
from utils import *

dataset = 'fake'

articles, labels = load_news_train_data(dataset)

train_size = int(cs.DATASET_MAX[dataset] * cs.TRAINING_PORTION)

Y_train = np.array(labels[:train_size])
Y_validate = np.array(labels[train_size:cs.DATASET_MAX[dataset]])

# fixme bag of words - implemented, but bugged
# naive_bayes(articles, labels, dataset)

# word2vec
X_train, X_validate = convert_to_word2vec(articles, dataset)

print(X_train[0])
# LSTM classifier
# LSTM_Classifier.lstm_classifier(X_train, X_validate, Y_train, Y_validate, dataset)

# Shallow word-level convolutional network
ShallowCNN.shallow_cnn_classifier(X_train, X_validate, Y_train, Y_validate, dataset)
# Deep  character-level  convolutional  networks

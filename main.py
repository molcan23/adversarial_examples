from naive_bayes import naive_bayes
from lstm_classifier import LSTM_Classifier
from shallow_CNN import ShallowCNN
from deep_CNN import DeepCNN
from utils import *

dataset = 'fake'


# Naive Bayes

# articles, labels = load_news_train_data(dataset)
#
# train_size = int(cs.DATASET_MAX[dataset] * cs.TRAINING_PORTION)
# Y_train = np.array(labels[:train_size])
# Y_validate = np.array(labels[train_size:cs.DATASET_MAX[dataset]])


# naive_bayes(articles, labels, dataset)

X_train, X_validate, Y_train, Y_validate = load_data(dataset)


# LSTM classifier
# LSTM_Classifier.lstm_classifier(X_train, X_validate, Y_train, Y_validate, dataset)


# Shallow word-level convolutional network
ShallowCNN.shallow_cnn_classifier(X_train, X_validate, Y_train, Y_validate, dataset)
# shallow_cnn_classifier(X_train, X_validate, Y_train, Y_validate, dataset)

# Deep  character-level  convolutional  networks
# DeepCNN.deep_cnn_classifier(X_train, X_validate, Y_train, Y_validate, dataset)

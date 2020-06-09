from lstm_classifier import LSTM_Classifier
from shallow_CNN import ShallowCNN
from deep_CNN import DeepCNN
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

# if not path.exists('X_train_' + dataset + '.csv') or not path.exists('X_validate_' + dataset + '.csv'):
#     DATA LOAD

# fixme lepsi sposob ulozenia
# save_w2v_text('X_train_' + dataset + '.csv', X_train)
# save_w2v_text('X_validate_' + dataset + '.csv', X_validate)

# else:
# default dtype for  np.loadtxt is also floating point, change it, to be able to load mixed data.
# X_train = load_w2v_text('X_train_' + dataset + '.csv')
# X_validate = load_w2v_text('X_validate_' + dataset + '.csv')
X_train, X_validate = convert_to_word2vec(articles, dataset)

print(X_train[0].shape)
print(X_validate[0].shape)
# LSTM classifier
# LSTM_Classifier.lstm_classifier(X_train, X_validate, Y_train, Y_validate, dataset)

# TODO zmenit Y triedu na 2D kde 1 = [0,1], 0 = [1, 0] - MOZNO

Y_train = expand_for_softmax(Y_train)
Y_validate = expand_for_softmax(Y_validate)

# Shallow word-level convolutional network
# ShallowCNN.shallow_cnn_classifier(X_train, X_validate, Y_train, Y_validate, dataset)

# Deep  character-level  convolutional  networks
DeepCNN.deep_cnn_classifier(X_train, X_validate, Y_train, Y_validate, dataset)

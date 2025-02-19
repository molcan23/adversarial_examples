from naive_bayes import naive_bayes
from lstm_classifier import LSTM_Classifier
from shallow_CNN import ShallowCNN
from deep_CNN import DeepCNN
from utils import *
from adversarial_examples import AdversarialExmaples


dataset = 'yelp'
# dataset = 'fake'

X_train, X_test, Y_train, Y_test, embedding, vocabulary = load_data(dataset)
articles, labels = load_bayes_train_data(dataset)

# testing set for adversarial examples
train_size = int(cs.DATASET_MAX[dataset] * cs.TRAINING_PORTION)
X_test_a = articles[train_size: cs.DATASET_MAX[dataset]]
Y_test_a = np.array(labels[train_size:cs.DATASET_MAX[dataset]])
X_train_a = articles[:train_size]
Y_train_LSTM = np.array([0 if i[0] == 1 else 1 for i in Y_train])
Y_test_LSTM = np.array([0 if i[0] == 1 else 1 for i in Y_test])


# Naive Bayes
# bayes_classifier, bayes_bag = naive_bayes(articles, labels, dataset)
#
# bayes_adversarial = AdversarialExmaples({'model': None, 'weights': None},
#                                         X_test_a, Y_test_a, X_train_a, bayes_classifier, bayes_bag, bayes=True)


# LSTM classifier
# lstm = LSTM_Classifier()
# lstm.lstm_classifier(X_train, X_test, Y_train_LSTM, Y_test_LSTM, dataset)
# lstm_classifier = lstm.model
#
# lstm_adversarial = AdversarialExmaples(embedding, X_test_a, Y_test_a, lstm_classifier)
# lstm_adversarial.evaluation()


# Shallow word-level convolutional network
shallow = ShallowCNN()
shallow.shallow_cnn_classifier(X_train, X_test, Y_train, Y_test, dataset)
shallow_classifier = shallow.model
#
# shallow_adversarial = AdversarialExmaples(embedding, X_test_a, Y_test_a, X_train, shallow_classifier, vocabulary)
# shallow_adversarial.evaluation()


# Deep  character-level  convolutional  networks
# deep = DeepCNN()
# deep.deep_cnn_classifier(X_train, X_test, Y_train, Y_test, dataset)
# deep_classifier = deep.model
#
# deep_adversarial = AdversarialExmaples(embedding, X_test_a, Y_test_a, deep_classifier)
# deep_adversarial.evaluation()

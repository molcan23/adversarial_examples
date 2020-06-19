from naive_bayes import naive_bayes
from lstm_classifier import LSTM_Classifier
from shallow_CNN import ShallowCNN
from deep_CNN import DeepCNN
from utils import *
from adversarial_examples import AdversarialExmaples  # , AdversarialBayes

dataset = 'yelp'
# dataset = 'fake'

X_train, X_validate, Y_train, Y_validate, embedding, vocabulary = load_data(dataset)

# print(Y_train)
# print(X_train[:2])

# Naive Bayes

articles, labels = load_bayes_train_data(dataset)
train_size = int(cs.DATASET_MAX[dataset] * cs.TRAINING_PORTION)

# original training text for NNs, needed for training language model
X_test = articles[train_size: cs.DATASET_MAX[dataset]]
Y_test = np.array(labels[train_size:cs.DATASET_MAX[dataset]])

# bayes_classifier = naive_bayes(articles, labels, dataset)
# bayes_adversarial = AdversarialBayes(bag, X_test, Y_test, bayes_classifier)


# LSTM classifier
# Y_train_LSTM = np.array([0 if i[0] == 1 else 1 for i in Y_train])
# Y_validate_LSTM = np.array([0 if i[0] == 1 else 1 for i in Y_validate])
#
# lstm = LSTM_Classifier()
# lstm.lstm_classifier(X_train, X_validate, Y_train_LSTM, Y_validate_LSTM, dataset)
# lstm_classifier = lstm.model
#
# lstm_adversarial = AdversarialExmaples(embedding, X_test, Y_test, lstm_classifier)
# lstm_adversarial.evaluation()

# Shallow word-level convolutional network
shallow = ShallowCNN()
shallow.shallow_cnn_classifier(X_train, X_validate, Y_train, Y_validate, dataset)
shallow_classifier = shallow.model
#
shallow_adversarial = AdversarialExmaples(embedding, X_test, Y_test, X_train, shallow_classifier, vocabulary)
print('aaaaaa')
# shallow_adversarial.evaluation()

# shallow_cnn_classifier(X_train, X_validate, Y_train, Y_validate, dataset)

# Deep  character-level  convolutional  networks
# deep = DeepCNN()
# deep.deep_cnn_classifier(X_train, X_validate, Y_train, Y_validate, dataset)
# deep_classifier = deep.model
#
# deep_adversarial = AdversarialExmaples(embedding, X_test, Y_test, deep_classifier)
# deep_adversarial.evaluation()

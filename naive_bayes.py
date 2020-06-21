from sklearn.naive_bayes import MultinomialNB
from utils import *


# convert each article into a bag-of-words representation
# binarize the word features and use a multinomial model for classification


def naive_bayes(articles, labels, dataset):
    train_size = int(cs.DATASET_MAX[dataset] * cs.TRAINING_PORTION)

    Y_train = np.array(labels[0: train_size])
    Y_test = np.array(labels[train_size:cs.DATASET_MAX[dataset]])

    X_train, X_test, bag = convert_to_bag_of_words_format(articles, dataset)

    gnb = MultinomialNB()
    model = gnb.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    print("Mislabeled:", (100 * (Y_test != y_pred).sum()) / X_test.shape[0], "%")

    return model, bag
    # adversarial podla cosinusovej vzdialenosti vektorov (bags cosine distance)

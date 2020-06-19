from sklearn.naive_bayes import MultinomialNB
from utils import *


# convert each article into a bag-of-words representation
# binarize the word features and use a multinomial model for classification


def naive_bayes(articles, labels, dataset):
    train_size = int(cs.DATASET_MAX[dataset] * cs.TRAINING_PORTION)

    Y_train = np.array(labels[0: train_size])
    Y_validate = np.array(labels[train_size:cs.DATASET_MAX[dataset]])
    # print(sum(Y_validate), len(Y_validate))

    X_train, X_validate = convert_to_bag_of_words_format(articles, dataset)

    [print(i) for i in X_train]
    print(X_validate)
    gnb = MultinomialNB()

    # test na natrenovani na sebe samom funguje
    # y_pred = gnb.fit(X_train, Y_train).predict(X_train)
    model = gnb.fit(X_train, Y_train)
    y_pred = model.predict(X_validate)

    # print(len(y_pred), Y_validate.shape)
    # print("Mislabeled:", (100 * (Y_train != y_pred).sum()), "%")
    print("Mislabeled:", (100 * (Y_validate != y_pred).sum()) / X_validate.shape[0], "%")

    return model
    # adversarial podla cosinusovej vzdialenosti vektorov (bags cosine distance)

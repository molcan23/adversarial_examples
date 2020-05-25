from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from sklearn.linear_model import LogisticRegression
import warnings
from os import path


# LSTM
#  single-layer 512 hidden neurons
# word2vec embedding of size 300
# mean of the outputs of all LSTM cells to form a feature vector,
# and then using multinomial logistic regression on this feature vector.
# The output dimensionis  512. The  variant  of  LSTM  we  used  is  the  common “vanilla”
# We also used gradient clipping in which the gradient norm is limited to 5

# We then average the outputs of the LSTMat each time step to obtain
# a feature vector for a final logistic regression to predict the sentiment.


def lstm_classifier(X_train, X_validate, Y_train, Y_validate, dataset):
    warnings.filterwarnings(action='ignore')
    # nltk.download('punkt')

    # if not path.exists('X_train_' + dataset + '.csv') or not path.exists('X_validate_' + dataset + '.csv'):
    #     DATA LOAD

    # fixme lepsi sposob ulozenia
    # save_w2v_text('X_train_' + dataset + '.csv', X_train)
    # save_w2v_text('X_validate_' + dataset + '.csv', X_validate)

    # else:
    # default dtype for  np.loadtxt is also floating point, change it, to be able to load mixed data.
    # X_train = load_w2v_text('X_train_' + dataset + '.csv')
    # X_validate = load_w2v_text('X_validate_' + dataset + '.csv')

    if not path.exists("model" + dataset + ".json") or not path.exists("model" + dataset + ".h5"):
        model = Sequential()
        model.add(LSTM(512))

        # FIXME ako urobit toto?
        # We then average the outputs of the LSTMat each time step to obtain
        # a feature vector for a final logistic regression to predict the sentiment

        # model.add(LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print(model.summary())
        model.fit(X_train, Y_train, validation_data=(X_validate, Y_validate), epochs=10, batch_size=64)

        model_json = model.to_json()
        with open("model" + dataset + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model" + dataset + ".h5")
        print("Saved model to disk")

        scores = model.evaluate(X_validate, Y_validate, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
    else:
        json_file = open("model_" + dataset + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model" + dataset + ".h5")
        print("Loaded model from disk")

        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        score = loaded_model.evaluate(X_validate, Y_validate, verbose=0)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))


# clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
# model = Sequential()
#
# # Masking layer for pre-trained embeddings
# model.add(Masking(mask_value=0.0))
#
# # Recurrent layer
# model.add(LSTM(64, return_sequences=False,
#                dropout=0.1, recurrent_dropout=0.1))
#
# # Fully connected layer
# model.add(Dense(64, activation='relu'))
#
# # Dropout for regularization
# model.add(Dropout(0.5))
#
# # Output layer
# model.add(Dense(num_words, activation='softmax'))



if __name__ == '__main__':
    lstm_classifier('fake')

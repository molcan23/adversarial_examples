from keras.models import Sequential, model_from_json
from keras.layers import Dense, GlobalAveragePooling1D, LSTM, Dropout, Embedding, Flatten
from keras import optimizers
from keras.callbacks import Callback
import warnings
from os import path
import global_variables as gv
import constants as cs


# LSTM
#  single-layer 512 hidden neurons
# word2vec embedding of size 300
# mean of the outputs of all LSTM cells to form a feature vector,
# and then using multinomial logistic regression on this feature vector.
# The output dimensionis  512. The  variant  of  LSTM  we  used  is  the  common “vanilla”
# We also used gradient clipping in which the gradient norm is limited to 5
# We then average the outputs of the LSTMat each time step to obtain
# a feature vector for a final logistic regression to predict the sentiment.

batch_size = 64


class OutputObserver(Callback):
    """
    callback to observe the output of the network
    """

    def __init__(self, X_train):
        self.out_log = []
        self.X_train = X_train

    def on_epoch_end(self, epoch, logs={}):
        print(self.model.predict(self.X_train, batch_size=batch_size)[:10])
        self.out_log.append(self.model.predict(self.X_train, batch_size=batch_size))


class LSTM_Classifier:

    def __init__(self):
        self.model = None

    def lstm_classifier(self, X_train, X_test, Y_train, Y_test, dataset):
        warnings.filterwarnings(action='ignore')
    
        if not path.exists("models/model_lstm_" + dataset + ".json") or \
                not path.exists("models/model_lstm_" + dataset + ".h5"):
    
            model = Sequential()
            model.add(LSTM(512, return_sequences=True, dropout=.5, recurrent_dropout=.5, bias_regularizer='l2'))
            model.add(GlobalAveragePooling1D())
            model.add(Dense(1, activation='sigmoid'))
            optimizer = optimizers.Adam(lr=.001, decay=0, clipnorm=5)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            output_observer = OutputObserver(X_train)

            model.fit(X_train, Y_train, validation_split=.2, epochs=5, batch_size=batch_size,
                      callbacks=[output_observer])

            print(output_observer.out_log)
    
            model_json = model.to_json()
            with open("models/model_lstm_" + dataset + ".json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("models/model_lstm_" + dataset + ".h5")
            print("Saved model to disk")
    
            scores = model.evaluate(X_test, Y_test, verbose=0)
            print("Accuracy: %.2f%%" % (scores[1] * 100))
            self.model = model
        else:
            json_file = open("models/model_lstm_" + dataset + ".json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("models/model_lstm_" + dataset + ".h5")
            print("Loaded model from disk")
    
            loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            print('test', loaded_model.predict(X_train))
            score = loaded_model.evaluate(X_test, Y_test, verbose=0)
            print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
            self.model = loaded_model
    # TODO natrenovane LSTM stale dava vysledok cca rovnake... -> opravit

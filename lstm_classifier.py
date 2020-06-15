from keras.layers import LSTM
from keras.models import Sequential, model_from_json
from keras.layers import Dense, GlobalAveragePooling1D
from keras import optimizers
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

class LSTM_Classifier:
    
    @staticmethod
    def lstm_classifier(X_train, X_validate, Y_train, Y_validate, dataset):
        warnings.filterwarnings(action='ignore')
    
        if not path.exists("models/model_lstm_" + dataset + ".json") or not path.exists("models/model_lstm_" + dataset + ".h5"):
    
            model = Sequential()
            model.add(LSTM(512, return_sequences=True))
            model.add(GlobalAveragePooling1D())
    
            model.add(Dense(1, activation='sigmoid'))

            adam = optimizers.Adam(lr=0.01, clipnorm=5)
            model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
            # print(model.summary())
            model.fit(X_train, Y_train, validation_data=(X_validate, Y_validate), epochs=10, batch_size=64)
    
            model_json = model.to_json()
            with open("models/model_lstm_" + dataset + ".json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("models/model_lstm_" + dataset + ".h5")
            print("Saved model to disk")
    
            scores = model.evaluate(X_validate, Y_validate, verbose=0)
            print("Accuracy: %.2f%%" % (scores[1] * 100))
        else:
            json_file = open("models/model_lstm_" + dataset + ".json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("models/model_lstm_" + dataset + ".h5")
            print("Loaded model from disk")
    
            loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            print('test', loaded_model.predict(X_train))
            score = loaded_model.evaluate(X_validate, Y_validate, verbose=0)
            print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
    # TODO natrenovane LSTM stale dava vysledok cca 0.4491... -> opravit

# # Recurrent layer
# model.add(LSTM(64, return_sequences=False,
#                dropout=0.1, recurrent_dropout=0.1))

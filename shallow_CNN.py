import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D
from os import path

np.random.seed(0)

# Model Hyperparameters
embedding_dim = 300
filter_size = 3
num_filters = 100
dropout_prob = 0.5
hidden_dims = 50

# Training parameters
batch_size = 30
num_epochs = 10


class ShallowCNN:

    def __init__(self):
        self.model = None

    def shallow_cnn_classifier(self, X_train, X_test, Y_train, Y_test, dataset):
        if not path.exists("models/model_shallow_cnn_" + dataset + ".json") or \
                not path.exists("models/model_shallow_cnn_" + dataset + ".h5"):

            model = Sequential()

            model.add(Convolution1D(filters=num_filters,
                                    kernel_size=filter_size,
                                    padding="valid",
                                    activation="relu",
                                    strides=1))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())

            model.add(Dropout(dropout_prob))
            model.add(Dense(hidden_dims, activation="relu"))
            model.add(Dense(2, activation="softmax"))

            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

            model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs,
                      validation_split=.2, verbose=2)

            # Save model
            model_json = model.to_json()
            with open("models/model_shallow_cnn_" + dataset + ".json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("models/model_shallow_cnn_" + dataset + ".h5")
            print("Saved model to disk")

            scores = model.evaluate(X_test, Y_test, verbose=0)
            print("Accuracy: %.2f%%" % (scores[1] * 100))

            self.model = model
        else:
            # Load model
            json_file = open("models/model_shallow_cnn_" + dataset + ".json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("models/model_shallow_cnn_" + dataset + ".h5")
            print("Loaded model from disk")

            loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            score = loaded_model.evaluate(X_test, Y_test, verbose=0)
            print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
            self.model = loaded_model

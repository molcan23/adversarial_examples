"""
Train convolutional network for sentiment analysis:
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim

embedding_dim = 300
filter_size = 3
num_filters = 100
dropout_prob = 0.5
hidden_dims = 50
- 1 filter size instead of original 3
- sliding Max Pooling instead of original Global Pooling  # fixme
"""

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
hidden_dims = 50  # fixme aky pocet?

# Training parameters
batch_size = 30
num_epochs = 10


class ShallowCNN:

    def __init__(self):
        self.model = None

    def shallow_cnn_classifier(self, X_train, X_validate, Y_train, Y_validate, dataset):
        if not path.exists("models/model_shallow_cnn_" + dataset + ".json") or \
                not path.exists("models/model_shallow_cnn_" + dataset + ".h5"):

            # Build model
            # sequence_length = X_train.shape[1]
            model = Sequential()
            # model.add(Input((sequence_length, embedding_dim)))

            # Convolutional block
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

            # Train the model
            model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs,
                      validation_data=(X_validate, Y_validate), verbose=2)

            # Save model
            model_json = model.to_json()
            with open("models/model_shallow_cnn_" + dataset + ".json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("models/model_shallow_cnn_" + dataset + ".h5")
            print("Saved model to disk")

            scores = model.evaluate(X_validate, Y_validate, verbose=0)
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
            score = loaded_model.evaluate(X_validate, Y_validate, verbose=0)
            print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
            self.model = loaded_model

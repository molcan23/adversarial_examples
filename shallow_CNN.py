from keras.models import Sequential, model_from_json
from keras.layers import Dense, GlobalMaxPooling1D, Conv2D, Reshape, Dropout
import warnings
from os import path


# Shallow word-level convolutional networks. An alternative approach to text classification are convolutional
# neural networks (CNNs; Kim, 2014)

# We train a CNN with an embedding layer (asin the LSTM)
# a temporal convolutional layer,
# followed by max-pooling over time,
# and a fully connected layer for classification.
# We use a uniform (ziaden logaritmicky) filter
# TODO size of 3 in each convolutional feature map;
#  1. filter size of 3 -> pre tri slova, cize 3*300?
#  2. each convolutional feature map -> pouziva sa viacero filtorv (1 filter = 1 feature map), ale ako pouzit viacero?
#   pouzit Conv3D? Neda sa to len tak priamo (urcite ale existuje nejaka funkcie na to - POZRIET) - musi sa "namnozit"
#   dane okno, vytvorime 3D vstup pre Conv3D (prvotny guess)
# all other settings are identical to those of Kim (2014):

# For regularization we employ dropout on the penultimate layer with a constraint on l2-norms of the weight vectors
# neviem ci spravne chapem - ze predposledna vrstva je dropout a s l2 regularizaciou?

# V Kimovi pise:
# For all datasets we use: rectified linear units, - kde pouzivaju ReLU? na output layer davaju softmax (pisu to)
# filter windows (h) of 3, 4, 5 with 100 feature maps each, - u nas to je 3x300
# dropout rate (p) of 0.5,
# l2 constraint (s) of 3, - TODO
# and mini-batch size of 50.
# These values were chosen via a grid search on the SST-2 dev set.

# Training is done through stochastic gradient descent over shuffled mini-batches
# with the Adadelta update rule (Zeiler, 2012). - je forma GD


class ShallowCNN:

    @staticmethod
    def shallow_cnn_classifier(X_train, X_validate, Y_train, Y_validate, dataset):
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

        if not path.exists("model_shallow_cnn" + dataset + ".json") or \
                not path.exists("model_shallow_cnn" + dataset + ".h5"):

            model = Sequential()

            # FIXME aby sa aplikovalo viacero filtrov (hore rozpisane co presne)
            model.add(Conv2D(kernel_size=(3, 300)))

            # max-pooling over time
            model.add(Reshape((X_train[1] * X_train[2], 1)))

            model.add(GlobalMaxPooling1D())
            model.add(Dropout(0.5, W_regularizer='l2'))

            model.add(Dense(2, activation='softmax'))
            model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
            # print(model.summary())
            model.fit(X_train, Y_train, validation_data=(X_validate, Y_validate), epochs=10, batch_size=50)

            model_json = model.to_json()
            with open("model_shallow_cnn" + dataset + ".json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("model_shallow_cnn" + dataset + ".h5")
            print("Saved model to disk")

            scores = model.evaluate(X_validate, Y_validate, verbose=0)
            print("Accuracy: %.2f%%" % (scores[1] * 100))
        else:
            json_file = open("model_shallow_cnn_" + dataset + ".json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("model_shallow_cnn" + dataset + ".h5")
            print("Loaded model from disk")

            loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            score = loaded_model.evaluate(X_validate, Y_validate, verbose=0)
            print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

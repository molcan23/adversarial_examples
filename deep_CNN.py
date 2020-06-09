from keras.models import Sequential, model_from_json
from keras.layers import Dense, GlobalMaxPooling1D, Conv2D, Reshape, Dropout, MaxPooling2D
import warnings
from os import path

import keras, os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Deep  character-level  CNN. We  implement  the  character-level  network  of Conneau et al. (2016),
# which includes 4 stages.  Each stage has 2 convolutional layers with batch normalization and 1 max-pooling layer;
# convolutional and pooling layers have strides of 1 and 2,respectively and filters of size 3.  We start
# with 64 feature maps, and double the amount after each pooling step, concluding with k-max pooling layer with k= 8.
# The resulting activations in R4096 are classified by 3 fully connected layers.

# https://arxiv.org/pdf/1606.01781.pdf strana 4


from keras.engine import Layer, InputSpec
from keras.layers import Flatten
import tensorflow as tf


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)


class DeepCNN:

    @staticmethod
    def deep_cnn_classifier(X_train, X_validate, Y_train, Y_validate, dataset):
        warnings.filterwarnings(action='ignore')

        # FIXME spravne pozmenenie siete z image recognitionu na NLP problem
        X_train = X_train.reshape(X_train.shape + (1,))
        X_validate = X_validate.reshape(X_validate.shape + (1,))

        if not path.exists("model_shallow_cnn" + dataset + ".json") or \
                not path.exists("model_shallow_cnn" + dataset + ".h5"):

            # model = Sequential()
            #
            # # We first apply one layer of 64 convolutions  of size 3, followed by a stack of temporal
            # # “convolutional blocks”.
            # model.add(Conv2D(kernel_size=(3, 64)))
            #
            # for i in range(4):
            #     model.add(Conv2D(kernel_size=(3, 64 * (2 ** i))))  # stride 1
            #     model.add(Conv2D(kernel_size=(3, 64 * (2 ** i))))  # stride 1
            #     # temporal batch  normalization  after  convolutional  layers  to regularize our network.
            #
            #     if i < 3:
            #         # (i) for the same output temporal resolution, the layers have the same number of feature maps,
            #         # (ii) when the temporal resolution  is halved,  the number  of feature  maps  is  doubled. - TODO
            #         # stackujeme feature mapy a vytvarame feature bloky
            #
            #         # The output of these convolutional blocks is a tensor of size 512×sd, where sd=s/2^p, p= 3
            #         # the number of down-sampling operations.
            #         # s = 1024 a je to # characters
            #         model.add(MaxPooling2D(pool_size=(2, 2)))  # FIXME pool size zmenit # stride 2
            #     else:
            #         model.add(KMaxPooling(k=8))
            #
            # model.add(Dense(2048, activation='ReLU'))
            # model.add(Dense(2048, activation='ReLU'))
            # model.add(Dense(2, activation='softmax'))  # todo softmax?

            # https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
            # hovory o architekture VDCNN

            # model = Sequential()
            # model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
            # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
            # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
            # model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
            # model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
            # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
            # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
            # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
            # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
            # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
            # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
            # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
            # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
            # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
            # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
            # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
            # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
            # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

            input_shape = X_train[0].shape

            model = Sequential()
            model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same",
                             activation="relu", strides=1))
            model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", strides=1))
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
            # rozdiel medzi max pool a max pooling
            # padding="same"?

            model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
            model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
            model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

            # FIXME ValueError: Input 0 is incompatible with layer k_max_pooling_1: expected ndim=3, found ndim=4
            #  pretoze to pouzival na obrazky a mal tak 3 dim (w,h, channels), tu mame len 2, cize jednu umelo pridame
            #  ale potom nefunguje najdena funkcia k max poolingu
            # model.add(KMaxPooling(k=8))

            model.add(Flatten())
            # bez flatten: ValueError: Error when checking target: expected dense_3 to have 4 dimensions,
            # but got array with shape (2, 1)
            # pouzite flatten: ValueError: Error when checking target: expected dense_3 to have shape (2,)
            # but got array with shape (1,)

            # zmenene Y na 2D kde 1 = [0,1], 0 = [1, 0]
            # TypeError: 'tuple' object is not an iterator

            model.add(Dense(units=2048, activation="relu"))
            model.add(Dense(units=2048, activation="relu"))
            model.add(Dense(units=2, activation="softmax"))

            opt = Adam(lr=0.001)
            model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

            checkpoint = ModelCheckpoint("vgg16_1.h5",
                                         monitor='val_acc', verbose=1,
                                         save_best_only=True,
                                         save_weights_only=False,
                                         mode='auto', period=1)
            early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
            hist = model.fit_generator(steps_per_epoch=100, generator=(X_train, Y_train),
                                       validation_data=(X_validate, Y_validate), validation_steps=10,
                                       epochs=100, callbacks=[checkpoint, early])

            plt.plot(hist.history["acc"])
            plt.plot(hist.history['val_acc'])
            plt.plot(hist.history['loss'])
            plt.plot(hist.history['val_loss'])
            plt.title("model accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
            plt.show()

            # print(model.summary())

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

import warnings
from os import path
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.backend.tensorflow_backend import set_session
from keras.engine import Layer, InputSpec
import tensorflow as tf


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        # k povodne rovne 3
        self.input_spec = InputSpec(ndim=4)
        self.k = k

    def compute_output_shape(self, input_shape):
        return input_shape[0], (input_shape[2] * self.k)

    def call(self, inputs):
        print(inputs)
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 1, 3, 2])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)


class DeepCNN:

    def __init__(self):
        self.model = None

    def deep_cnn_classifier(self, X_train, X_test, Y_train, Y_test, dataset):
        warnings.filterwarnings(action='ignore')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        sess = tf.Session(config=config)
        set_session(sess)

        X_train = X_train.reshape(X_train.shape + (1,))
        X_test = X_test.reshape(X_test.shape + (1,))

        if not path.exists("model_shallow_cnn" + dataset + ".json") or \
                not path.exists("model_shallow_cnn" + dataset + ".h5"):

            tf.reset_default_graph()
            input_shape = X_train[0].shape

            model = Sequential()
            model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same",
                             activation="relu", strides=1))
            model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", strides=1))
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

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

            model.add(KMaxPooling(k=8))

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

            # Resource exhausted: OOM when allocating tensor with shape[90,3438,300,64]  # este pre fake_news

            # FIXME
            model.fit(steps_per_epoch=1, x=X_train, y=Y_train,
                      validation_split=.2, validation_steps=10,
                      epochs=10, callbacks=[checkpoint, early])

            model_json = model.to_json()
            with open("model_shallow_cnn" + dataset + ".json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("model_shallow_cnn" + dataset + ".h5")
            print("Saved model to disk")

            scores = model.evaluate(X_test, Y_test, verbose=0)
            print("Accuracy: %.2f%%" % (scores[1] * 100))
            self.model = model
        else:
            json_file = open("model_shallow_cnn_" + dataset + ".json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("model_shallow_cnn" + dataset + ".h5")
            print("Loaded model from disk")

            loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            score = loaded_model.evaluate(X_test, Y_test, verbose=0)
            print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
            self.model = loaded_model

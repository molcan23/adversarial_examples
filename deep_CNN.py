from keras.models import Sequential, model_from_json
from keras.layers import Dense, GlobalMaxPooling1D, Conv2D, Reshape, Dropout, MaxPooling2D
import warnings
from os import path

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

            # We first apply one layer of 64 convolutions  of size 3, followed by a stack of temporal
            # “convolutional blocks”.
            model.add(Conv2D(kernel_size=(3, 64)))

            for i in range(4):
                model.add(Conv2D(kernel_size=(3, 64 * (2 ** i))))
                model.add(Conv2D(kernel_size=(3, 64 * (2 ** i))))
                # temporal batch  normalization  after  convolutional  layers  to regularize our network.

                if i < 3:
                    # (i) for the same output temporal resolution, the layers have the same number of feature maps,
                    # (ii) when the temporal resolution  is halved,  the number  of feature  maps  is  doubled. - TODO
                    # stackujeme feature mapy a vytvarame feature bloky

                    # The output of these convolutional blocks is a tensor of size 512×sd, where sd=s/2^p, p= 3
                    # the number of down-sampling operations.
                    # s = 1024 a je to # characters
                    model.add(MaxPooling2D(pool_size=(2, 2)))  # FIXME pool size zmenit
                else:
                    model.add(KMaxPooling(k=8))

            model.add(Dense(2048, activation='ReLU'))
            model.add(Dense(2048, activation='ReLU'))
            model.add(Dense(2, activation='softmax'))   # todo softmax?

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

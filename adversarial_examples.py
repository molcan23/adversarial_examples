import numpy as np
import constants as cs


# TODO ako brat susedov pre naive bayes?

class AdversarialExmaples:

    def __init__(self, embedding, X_test, Y_test, target_classifier):
        """
        :param embedding: w2v model & weights
        :param X_test, Y_test: for training language model P
        :param target_classifier
        """
        self.embedding_model = embedding['model']
        self.embedding_weights = embedding['weights']
        self.X_test = X_test
        self.Y_test = Y_test
        self.target_classifier = target_classifier

        self.X_test_adversarial = None
        self.P = self.train_P()

    # TODO P
    def train_P(self):
        language_model = 0
        return language_model

    # SEMANTIC SIMILARITY
    # thought vectors that are averages of the vectors for individual words
    # ‖v−v′‖2< γ1
    def semantic_similarity(self, x, x_prime):
        return np.sqrt(np.sum(np.mean(self.embedding_weights(x), axis=1) -
                              np.mean(self.embedding_weights(x_prime), axis=1) ** 2)) <= cs.GAMA1

    # SYNTACTIC SIMILARITY
    # |logP(x′)−logP(x)|< γ2
    def syntactic_similarity(self, x, x_prime):
        return abs(np.log(self.P(x)) - np.log(self.P(x_prime))) < cs.GAMA2

    # TODO J
    # J(x′) measures the extent to which x′ is adversarial and may be a function of a target class y′!=y,
    # e.g.J(x′)  =fy′(x′)
    def J(self, x):
        return 0

    def algoritm1_greedy_opt_strategy_for_finding_adversarial_examples(self, x):
        """
        Algorithm 1 requires access to a target classifier f; it transforms x into x′by optimizing the objective J
        :param x: datapoint
        :param tau: termination threshold
        :param gama1
        :param gama2
        :param delta
        :return:
        """
        num_words = len(x)
        changed = 0
        x_prime = [a for a in x]

        while self.J(x_prime) < cs.TAU and (changed/num_words) < cs.DELTA:

            working_set = []
            for i in range(len(x)):
                x_stripe = [a for a in x]

                # To ensure that the replacements are also synonyms, we use the GloVE word vectors post-processed by
                # with the method of Mrksic et al. (2016) -> TODO
                for w_stripe in self.embedding_model.wv.most_similar([x[i]], topn=cs.NEIGHBORHOOD):
                    x_stripe[i] = w_stripe
                    if self.syntactic_similarity(x, x_stripe):
                        working_set.append([x_stripe, i])

            if not working_set:
                break

            max_J = float('-inf')
            for x_stripe in working_set:
                if self.semantic_similarity(x, x_stripe):
                    value_J = self.J(x_stripe)
                    if value_J > max_J:
                        max_J = value_J
                        x_prime = x_stripe

            changed += 1

        return x_prime

    def evaluation(self):
        score = self.target_classifier.evaluate(self.X_test_adversarial, self.Y_test, verbose=0)
        print("Adversarial %s: %.2f%%" % (self.target_classifier.metrics_names[1], score[1] * 100))


class AdversarialBayes:

    def __init__(self, bag, X_test, Y_test, target_classifier):
        """
        :param bag: w2v model & weights
        :param X_test, Y_test: for training language model P
        :param target_classifier
        """
        self.bag = bag
        self.X_test = X_test
        self.Y_test = Y_test
        self.target_classifier = target_classifier

        self.X_test_adversarial = None
        self.P = self.train_P()

    # TODO P
    def train_P(self):
        language_model = 0
        return language_model

    # SEMANTIC SIMILARITY
    # thought vectors that are averages of the vectors for individual words
    # ‖v−v′‖2< γ1
    def semantic_similarity(self, x, x_prime):
        return np.sqrt(np.sum(np.mean(self.embedding_weights(x), axis=1) -
                              np.mean(self.embedding_weights(x_prime), axis=1) ** 2)) <= cs.GAMA1

    # SYNTACTIC SIMILARITY
    # |logP(x′)−logP(x)|< γ2
    def syntactic_similarity(self, x, x_prime):
        return abs(np.log(self.P(x)) - np.log(self.P(x_prime))) < cs.GAMA2

    # TODO J
    # J(x′) measures the extent to which x′ is adversarial and may be a function of a target class y′!=y,
    # e.g.J(x′)  =fy′(x′)
    def J(self, x):
        return 0

    def algoritm1_greedy_opt_strategy_for_finding_adversarial_examples(self, x):
        """
        Algorithm 1 requires access to a target classifier f; it transforms x into x′by optimizing the objective J
        :param x: datapoint
        :param tau: termination threshold
        :param gama1
        :param gama2
        :param delta
        :return:
        """
        num_words = len(x)
        changed = 0
        x_prime = [a for a in x]

        while self.J(x_prime) < cs.TAU and (changed/num_words) < cs.DELTA:

            working_set = []
            for i in range(len(x)):
                x_stripe = [a for a in x]

                # To ensure that the replacements are also synonyms, we use the GloVE word vectors post-processed by
                # with the method of Mrksic et al. (2016) -> TODO
                for w_stripe in self.embedding_model.wv.most_similar([x[i]], topn=cs.NEIGHBORHOOD):
                    x_stripe[i] = w_stripe
                    if self.syntactic_similarity(x, x_stripe):
                        working_set.append([x_stripe, i])

            if not working_set:
                break

            max_J = float('-inf')
            for x_stripe in working_set:
                if self.semantic_similarity(x, x_stripe):
                    value_J = self.J(x_stripe)
                    if value_J > max_J:
                        max_J = value_J
                        x_prime = x_stripe

            changed += 1

        return x_prime

    def adversarial(self):
        return self.X_test_adversarial

    def evaluation(self):
        score = self.target_classifier.evaluate(self.X_test_adversarial, self.Y_test, verbose=0)
        print("Adversarial %s: %.2f%%" % (self.target_classifier.metrics_names[1], score[1] * 100))

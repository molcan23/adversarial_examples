from scipy import spatial
from utils import *
from sklearn.metrics.pairwise import cosine_similarity
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

# GloVe http://nlp.stanford.edu/data/glove.6B.zip
embeddings_dict = {}
with open("glove.6B.50d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


def find_closest_embeddings(word):
    embedding = embeddings_dict[word]
    return sorted(embeddings_dict.keys(),
                  key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))[:cs.GloVe_LEN]


class AdversarialExmaples:

    def __init__(self, embedding, X_test, Y_test, X_train, target_classifier, vocabulary, bayes=False):
        """
        :param embedding: w2v model & weights or empty dict for Bayes
        :param X_test, Y_test: change
        :param X_train: for training language model P
        :param target_classifier
        :param vocabulary - w2v vocabulary OR bag dict for Bayes
        :param bayes - bool - true if creates adversarial for Naive Bayes
        """
        self.embedding_model = embedding['model']
        self.embedding_weights = embedding['weights']
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_train = clean_text(X_train)
        self.target_classifier = target_classifier
        self.vocabulary = vocabulary
        self.bayes = bayes
        self.P = self.train_P()
        self.J = self.J_bayes if bayes else self.J_w2v
        self.semantic_similarity = self.semantic_similarity_bayes if bayes else self.semantic_similarity_w2w

        self.X_test_adversarial = []
        self.create_adversarial()

    def create_adversarial(self):
        for x, y in zip(self.X_test, self.Y_test):
            self.X_test_adversarial.append(self.algorithm1_greedy_opt_strategy_for_finding_adversarial_examples(x, y))

    def train_P(self):
        n = 3
        train_data, padded_sents = padded_everygram_pipeline(n, self.X_train)

        language_model = MLE(n)
        language_model.fit(train_data, padded_sents)
        language_model.vocab()
        return language_model

    # SEMANTIC SIMILARITY
    # thought vectors that are averages of the vectors for individual words
    # ‖v−v′‖2< γ1
    def semantic_similarity_w2w(self, x, x_prime):
        return np.sqrt(np.sum(np.mean(self.embedding_weights[x], axis=1) -
                              np.mean(self.embedding_weights[x_prime], axis=1) ** 2)) <= cs.GAMA1

    def semantic_similarity_bayes(self, x, x_prime):
        return cosine_similarity(bag_of_words([x])[1][0], bag_of_words([x_prime])[1][0]) <= cs.GAMA1

    # SYNTACTIC SIMILARITY
    # |logP(x′)−logP(x)|< γ2
    def syntactic_similarity(self, x, x_prime, changed_position):
        x = ' '.join([word for word in x])
        x_prime = ' '.join([word for word in x_prime])

        return abs(np.log(self.P.score(x[changed_position], x[changed_position-1].split()))
                   - np.log(self.P.score(x_prime[changed_position], x_prime[changed_position-1].split()))) < cs.GAMA2

    # J(x′) measures the extent to which x′ is adversarial and may be a function of a target class y′!=y
    def J_w2v(self, x, y):
        x = clean_str(x).split(" ")
        words = [w for w in x if w not in cs.STOP_WORDS]
        new_x = build_input_data([words], [], self.vocabulary)
        x = tf.keras.preprocessing.sequence.pad_sequences(new_x[0], maxlen=gv.max_article_len)
        x = np.stack([np.stack([self.embedding_weights[word] for word in sentence]) for sentence in x])
        # print(self.target_classifier.predict(x), abs(y - 1))
        return self.target_classifier.predict(x)[0][abs(y - 1)] < cs.TAU

    def J_bayes(self, x, y):
        # v tomto pripade to bude len binarne, leb naive bayes vracia len 0/1
        x = clean_str(x).split(" ")
        words = [w for w in x if w not in cs.STOP_WORDS]
        return self.target_classifier.predict(bag_of_words([words])[1])[0] == y

    def algorithm1_greedy_opt_strategy_for_finding_adversarial_examples(self, x, y):
        """
        Algorithm 1 requires access to a target classifier f; it transforms x into x′ by optimizing the objective J
        """

        x = clean_text([x])[0]
        num_words = len(x)
        changed = 0
        x_prime = [a.lower() for a in x]

        sentence = ' '.join([word for word in x_prime])
        while self.J(sentence, y) and (changed/num_words) < cs.DELTA:
            working_set = []
            for i in range(len(x)):
                x_stripe = [a for a in x_prime]
                if x_prime[i] not in embeddings_dict: continue
                closest_embeddings = find_closest_embeddings(x_prime[i])
                if not self.bayes:
                    if x_prime[i] not in self.embedding_model.wv: continue
                    most_similar = self.embedding_model.wv.most_similar([x_prime[i]], topn=cs.NEIGHBORHOOD)
                    for w_stripe in most_similar:
                        if w_stripe not in closest_embeddings: continue
                        x_stripe[i] = w_stripe
                        if self.syntactic_similarity(x_prime, x_stripe, i):
                            working_set.append([x_stripe, i])
                else:
                    if x_prime[i] not in self.vocabulary: continue
                    # TODO chcem vsetky bigramy s x'[i-1] aby som zmenil za x'[i] len za 'validne'
                    # zatial som nenasiel takuto funkciu
                    for key in self.vocabulary:
                        x_stripe[i] = key
                        if self.syntactic_similarity(x_prime, x_stripe, i):
                            working_set.append([x_stripe, i])

            if not working_set:
                break

            max_J = float('-inf')
            for x_stripe in working_set:
                bag_x_prime = bag_of_words([x_prime])[1][0]
                bag_x_stripe = bag_of_words([x_stripe])[1][0]
                if self.semantic_similarity(bag_x_prime, bag_x_stripe):
                    value_J = self.J(x_stripe, y)
                    if value_J > max_J:
                        max_J = value_J
                        x_prime = x_stripe

            changed += 1
            sentence = ' '.join([word for word in x_prime])
        return x_prime

    def evaluation(self):
        score = self.target_classifier.evaluate(self.X_test_adversarial, self.Y_test, verbose=0)
        print("Adversarial %s: %.2f%%" % (self.target_classifier.metrics_names[1], score[1] * 100))

from nltk.corpus import stopwords
import string


TABLE = str.maketrans('', '', string.punctuation)
STOP_WORDS = set(stopwords.words('english'))
TRAINING_PORTION = .9

EMBEDDING_DIM = 300
WINDOW = 3
MIN_COUNT = 5

DATASET_PATHS = {'fake': 'data/fake-news/train.csv',
                 'yelp': 'data/yelp-review-polarity/yelp_review_polarity_csv/train.csv'}
DATASET_NAMES = {'fake': 'fake', 'yelp': 'yelp'}
DATASET_MAX = {'fake': 30, 'yelp': 10}
# malo pamate uz aj pre 600, pre 500 horsia val acc ako pri 300 -> najvysia vsak je len 0.6667

EXP_SIZE = 100

# Word2Vec parameters (see train_word2vec)
MIN_WORD_COUNT = 1
CONTEXT = 10


# constants for adversarial examples alg
TAU = .7
GAMA1 = .2
GAMA2 = 2
DELTA = .5
NEIGHBORHOOD = 20

NGRAM_ORDER = 3
# # of similar words returned by find_closest_embeddings(word)
GloVe_LEN = 20

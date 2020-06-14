from nltk.corpus import stopwords
import string


TABLE = str.maketrans('', '', string.punctuation)
STOP_WORDS = set(stopwords.words('english'))
TRAINING_PORTION = .9

EMBEDDING_DIM = 300
WINDOW = 3
MIN_COUNT = 5

DATASET_PATHS = {'fake': 'data/fake-news/train.csv'}
DATASET_NAMES = {'fake': 'fake'}
DATASET_MAX = {'fake': 600}

EXP_SIZE = 100

# Word2Vec parameters (see train_word2vec)
MIN_WORD_COUNT = 1
CONTEXT = 10

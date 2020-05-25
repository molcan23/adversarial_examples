from nltk.corpus import stopwords
import string


TABLE = str.maketrans('', '', string.punctuation)
STOP_WORDS = set(stopwords.words('english'))
TRAINING_PORTION = .9

SIZE = 300
WINDOW = 3
MIN_COUNT = 5

DATASET_PATHS = {'fake': 'data/fake-news/train.csv'}
DATASET_NAMES = {'fake': 'fake'}
DATASET_MAX = {'fake': 600}

EXP_SIZE = 600

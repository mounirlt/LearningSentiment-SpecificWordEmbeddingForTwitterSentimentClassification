import numpy as np

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

class WordEmbeddingVectorizer(object):
    """docstring for WordEmbeddingVectorizer"""
    def __init__(self, model_library, size):
        self.model_library = model_library
        self.word_weight = None
        self.dim = size

    def set_model(self, model_library):
        self.model_library = model_library
        self.word_weight = None
        self.dim = len(model_library.itervalues().next())

    """Build TFIDF bags from train set text"""
    def fit(self, train_set):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(train_set)

        max_idf = max(tfidf.idf_)
        self.word_weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        self.feature_names = tfidf.get_feature_names()

        return self

    def transform(self, train_set):
        return np.array([
                np.sum([self.model_library[w]
                    for w in words if w in self.model_library] or
                    [np.zeros(self.dim)], axis=0)
                for words in train_set])

    def fit_transform(self, train_set):
        vect = self.fit(train_set)
        return vect.transform(train_set)

    def get_feature_names(self):
        return self.feature_names

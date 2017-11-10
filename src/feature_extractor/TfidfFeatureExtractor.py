from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from FeatureExtractor import FeatureExtractor


class TfidfFeatureExtractor(FeatureExtractor):
    """docstring for FeatureExtractor"""
    def __init__(self, dataset=None, size=0):
        super(TfidfFeatureExtractor, self).__init__(dataset)
        self.size = size
        self.reducer = None

    def build(self):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.dataset)

        # Build svd reducer
        if self.size > 0:
            self.reducer = TruncatedSVD(n_components=self.size)
        return self

    def extract_features(self, dataset):
        features = super(TfidfFeatureExtractor, self).extract_features(dataset)
        if self.reducer:
            return self.reducer.fit_transform(features)
        else:
            return features

    def extract_existing_features(self, dataset):
        features = super(TfidfFeatureExtractor, self).extract_features(dataset)
        if self.reducer:
            return self.reducer.transform(features)
        else:
            return features

    def get_name(self):
        return "Term Frequency - Inverse Document Frequency"

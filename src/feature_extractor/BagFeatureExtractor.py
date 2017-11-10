from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from FeatureExtractor import FeatureExtractor


class BagFeatureExtractor(FeatureExtractor):
    """docstring for FeatureExtractor"""
    def __init__(self, dataset=None, size=0):
        super(BagFeatureExtractor, self).__init__(dataset)
        self.size = size
        self.reducer = None

    def build(self):
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(self.dataset)

        # Build svd reducer
        if self.size > 0:
            self.reducer = TruncatedSVD(n_components=self.size)
        return self

    def extract_features(self, dataset):
        features = super(BagFeatureExtractor, self).extract_features(dataset)
        if self.reducer:
            return self.reducer.fit_transform(features)
        else:
            return features

    def extract_existing_features(self, dataset):
        features = super(BagFeatureExtractor, self).extract_features(dataset)
        if self.reducer:
            return self.reducer.transform(features)
        else:
            return features

    def get_name(self):
        return "Bag of Words"

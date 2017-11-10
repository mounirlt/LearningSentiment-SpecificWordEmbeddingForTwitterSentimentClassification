class FeatureExtractor(object):
	"""docstring for FeatureExtractor"""
	def __init__(self, dataset=None):
		if dataset:
			self.dataset = dataset
		self.vectorizer = None

	def set_dataset(self, dataset):
		self.dataset = dataset

	def extract_features(self, dataset):
		return self.vectorizer.transform(dataset)

	def get_feature_size(self):
		return len(self.vectorizer.get_feature_names())

	def save_vocab(self, outfile):
		with open(outfile, "wb") as vocabfile:
			for vocab in self.vectorizer.get_feature_names():
				vocabfile.write(vocab + "\n")

		return outfile

class WordEmbeddingAnalyzer(object):
    """docstring for WordEmbeddingAnalyzer"""
    def __init__(self, model_path=None, senna=0):
        super(WordEmbeddingAnalyzer, self).__init__()
        if model_path:
            self.model = model_path
            self.is_senna = False

    def load_model(self, model_path):
        pass

    def most_similar(self, word):
        pass

    def visualize(self):
        pass

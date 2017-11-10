import pandas as pd  # provide sql-like data manipulation tools. very handy.
from sklearn.manifold import TSNE
from gensim.models.keyedvectors import KeyedVectors
from bokeh.models import HoverTool
from bokeh.plotting import figure, show
import pprint

def WordVectorFileFormat(sizeOfEmbedding, VocabsFname, VectorsFname, wordVectorFileFormattedFname):
    """
    :return:
    :param sizeOfEmbedding: Size of the words embedding in the vectors file (200)
    :param VocabsFname: Filename where vocab words used for embedding are stored . txt Formatting of the file ('vocabs.txt')
    :param VectorsFname: Filename where vector embedding of the vocab words is located . txt Formatting of the file ('vectors.txt')
    :param wordVectorFileFormattedFname: Filename where WordVectorFile will be stored (  "w2vVectors_jdid.txt" )
    :return:

    """

    def file_len(fname):
        """
        :param fname: Filename of a file with a NON-zero number of rows
        :return: number of rows of the input file
        """
        ### TODO : set an exception when the file is empty.
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        f.close()
        return i + 1

    with open(VectorsFname, 'r') as vectorsFile:

        with open(VocabsFname, 'r') as vocabsFile:
            vocabLine = vocabsFile.readlines()
            vectorLine = vectorsFile.readlines()
            newline = list()
            for i in range(vocabLine.__len__()):
                newline.append(vocabLine[i].rstrip() + ' ' + vectorLine[i])
            # del newline[17915]

            with open(wordVectorFileFormattedFname, "w") as wordVectorFileFormatted:
                wordVectorFileFormatted.write(str(file_len(VocabsFname) - 1) + ' ' + str(sizeOfEmbedding) + '\n')
                for l in newline:
                    wordVectorFileFormatted.writelines(l)
            wordVectorFileFormatted.close()
        vocabsFile.close()
    vectorsFile.close()


class Visualiser():
    def __init__(self, sizeOfEmbedding=50, VocabsFname=None,
                 VectorsFname=None, WVFilename=None,
                 visualizerHTMLfilename=None):
        # attributes used for the WordVectorFileFormat function.
        self.sizeOfEmbedding = sizeOfEmbedding
        self.VocabsFname = VocabsFname
        self.VectorsFname = VectorsFname
        self.WVFilename = WVFilename
        self.visualizerHTMLfilename = visualizerHTMLfilename

    def visualize(self):
        # Constructing the wordvectors File from vectors.txt and vocabs.txt
        sizeOfEmbedding = self.sizeOfEmbedding
        VocabsFname = self.VocabsFname
        VectorsFname = self.VectorsFname
        WVFilename = self.WVFilename
        visualizerHTMLfilename = self.visualizerHTMLfilename
        WordVectorFileFormat(sizeOfEmbedding, VocabsFname, VectorsFname, WVFilename)

        # Save the vectors using : word_vectors = KeyedVectors.
        word_vectors = KeyedVectors.load_word2vec_format(fname=WVFilename, binary=False)

        # defining the chart
        visualizerHTMLfilename = open(visualizerHTMLfilename)
        # Configuring the figure parameters.
        plot_tfidf = figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",
                                           tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave", x_axis_type=None,
                                           y_axis_type=None,
                                           min_border=1)

        #most_similar_to_happy = word_vectors.most_similar('happy')
        #print("\n The most similar to happy are: ")
        #print(most_similar_to_happy)
        pp = pprint.PrettyPrinter(indent=4)
        var = raw_input("Please enter a word OR press q to quit: ")
        while var !="q":
            print("\n The most similar words using the cosine similarity measure to "+var+" are: ")
            most_sim = word_vectors.most_similar(var)
            pp.pprint(word_vectors.most_similar(var))
            var = raw_input("\n Please enter a word to compute its similarity OR press q to quit: ")
        print("\n The most similar words using the MCO measure to")
        var = raw_input("Please enter a word press q to quit: ")
        while var !="q":
            print("\n The most similar to "+var+" are: ")
            most_sim_cum = word_vectors.most_similar_cosmul(var)
            pp.pprint(word_vectors.most_similar_cosmul(var))
            var = raw_input("Please enter a word OR press q to quit: ")
        #most_similar_to_sad = word_vectors.most_similar('sad')
        #most_similar_to_ugly = word_vectors.most_similar('ugly')

        #  define the word_vectors to plot in the figure
        word_vectors_to_plot = [word_vectors.word_vec(w) for w in word_vectors.vocab.keys()[:5000]]
        # dimensionality reduction. converting the vectors to 2d vectors
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
        tsne_w2v = tsne_model.fit_transform(word_vectors_to_plot)

        # putting everything in a dataframe
        pd.options.mode.chained_assignment = None
        tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
        # tsne_df['words'] = tweet_w2v.wv.vocab.keys()[:5000]
        tsne_df['words'] = word_vectors.vocab.keys()[:5000]

        # plotting. the corresponding word appears when you hover on the data point.
        plot_tfidf.scatter(x='x', y='y', source=tsne_df)
        hover = plot_tfidf.select(dict(type=HoverTool))
        hover.tooltips = {"word": "@words"}
        show(plot_tfidf)

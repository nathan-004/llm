import numpy as np
import re

def enumerate(iterable):
    return [(i, iterable[i]) for i in range(len(iterable))]

class NLP():
    def __init__(self):
        pass

    def one_hot(self, corpus):
        """
        Returns a vocabulary like {"word": [0,0,0,1,0]}
        """
        voc = self.voc_creation(corpus)
        length = len(voc)

        for word in voc:
            voc[word] = [0 if i != voc[word] else 1 for i in range(length)]

        return voc

    def voc_creation(self, corpus):
        """
        Returns a vocabulary {"word": 0, "word2": 1}
        """
        corpus = self.clean_data(corpus)
        voc = {}

        word_i, phrase_i, i = 0,0,0

        while True:
            if not corpus[phrase_i][word_i] in voc:
                voc[corpus[phrase_i][word_i]] = i
                i+=1

            word_i += 1

            if word_i >= len(corpus[phrase_i]):
                word_i = 0
                phrase_i += 1
                if phrase_i >= len(corpus):
                    break

        return voc

    def clean_data(self, corpus):
        """
        Actions
        -------
        - Delete the punctuation

        s+ -> un ou plusieurs caractères d'espacement
        w -> caractère alphanumérique (lettres et chiffres)
        """

        for idx, text in enumerate(corpus):
            text = re.sub(r'[^\w\s]', '', text)
            corpus[idx] = text.split(" ")

        return corpus

class Word2Vec():
    def __init__(self):
        """
        V    Number of unique words in our corpus of text ( Vocabulary )
        x    Input layer (One hot encoding of our input word ).
        N    Number of neurons in the hidden layer of neural network
        W    Weights between input layer and hidden layer
        W'   Weights between hidden layer and output layer
        y    A softmax output layer having probabilities of every word in our vocabulary
        """

        self.N = 10
        self.X_train = []
        self.y_train = []
        self.window_size = 2
        self.alpha = 0.001 # ?
        self.words = []
        self.word_index = {}


    def initialize(self, V, data):
        """
        Initie des variables importantes à partir de données.

        Parameters
        ----------
        V:int
            Nombre de mots uniques dans le corpus
        data:dict
            Dictionnaire sous la forme {"mot": itérations de ce mot dans le corpus}
        """
        self.V = V  # Nombre de mots dans le vocabulaire
        self.W = np.random.uniform(-0.8, 0.8, (self.V, self.N))
        self.W1 = np.random.uniform(-0.8, 0.8, (self.N, self.V))
        self.words = data
        for i in range(len(data)):
            self.word_index[data[i]] = i




corpus = ["John likes to watch movies. Mary likes movies too.",
        "Mary also likes to watch football games."]

nlp = NLP()

print(nlp.one_hot(corpus))


# https://radimrehurek.com/gensim/models/word2vec.html
# https://www.baeldung.com/cs/convert-word-to-vector
# https://www.geeksforgeeks.org/implement-your-own-word2vecskip-gram-model-in-python/
# https://www.geeksforgeeks.org/continuous-bag-of-words-cbow-in-nlp/

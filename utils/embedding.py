class PretrainedWordEmbedding(object):
    def __init__(self, fname, word_index):
        """
        :param fname: 词向量路径
        :param word_index: Bow().word_index
        """
        self.word_index = word_index
        self.embeddings_index, self.embeddings_dim = self._load_wv(fname)

    @property
    def embedding_matrix(self):
        print('Embedding Matrix ...')
        # prepare embedding matrix
        num_words = len(self.word_index) + 1  # 未出现的词标记0
        embedding_matrix = np.zeros((num_words, self.embeddings_dim))
        # embedding_matrix = np.random.random((num_words, EMBEDDING_DIM))
        # np.random.normal(size=(num_words, EMBEDDING_DIM))
        for word, idx in self.word_index.items():
            if word in self.embeddings_index:
                # 不在词向量的词初始化为0或者其他
                embedding_matrix[idx] = self.embeddings_index[word]
        return embedding_matrix
        
        
    def _load_wv(self, fname):
        try:
            import gensim
            print('Load Word Vectors By gensim ...')
            model = gensim.models.KeyedVectors.load_word2vec_format(fname)
            return model, model.vector_size
        except ImportError:
            ImportError("Please install gensim")
            return self.__load_wv(fname)

    def __load_wv(self, fname):
        embeddings_index = {}
        with open(fname) as f:
            for line in tqdm(f, 'Load Word Vectors By open ...'):
                line = line.split()
                if len(line) > 2:
                    embeddings_index[line[0]] = np.asarray(line[1:], dtype='float32')
        return embeddings_index, len(line[1:])

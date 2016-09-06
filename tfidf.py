import numpy as np
from collections import Counter


class TFIDF:
    """
    Computes TFIDF given a set of words. Includes matching functions and other tricks.
    """
    def __init__(self, doc_names):
        # Initialize
        print "Initializing TFIDF for %d documents" %len(doc_names)
        self.doc_names = doc_names
        self.ndocs = len(doc_names)

    def get_filtered_vocabulary(self, all_words, word_thresh=1):
        """
        Get the list of words used in the documents and filter it based on word frequency

        Inputs:
            all_words: a document list of word list [['d1w1', 'd1w2'], ['d2w1', 'd2w2']]
            word_thresh: at least N words to add to vocabulary

        Outputs:
            vocab: a list of words
        """

        assert(len(all_words) == self.ndocs)

        # Get vocabulary
        allwordslist = [w for doc in all_words for w in doc]
        vocab = Counter(allwordslist)
        # filter vocabulary
        filt_vocab = { k:vocab[k] for k in vocab.keys() if vocab[k] >= word_thresh }
        # filt_vocab = vocab
        vocab = sorted(filt_vocab.keys())
        print "\nVocabulary length: " + str(len(vocab))

        self.vocab = vocab


    def compute_tfidf(self, all_words):
        """
        Compute TF-IDF weighting for the documents.
        Uses: tf(t, d) = (1 + log10(frequency(t, d)))
              idf(t) = log10(#documents / sum(document-has-term))

        Inputs:
            vocab: a list of words part of the vocabulary (see get_filtered_vocabulary)
            all_words: a document list of word list [['d1w1', 'd1w2'], ['d2w1', 'd2w2']]

        Outputs:
            tf: simple term frequencies dictionary
            idf: inverted frequency, before log
            tfidf: (1 + log(tf)) * log(idf)
        """

        assert(len(all_words) == self.ndocs)

        # Get term frequencies
        print "Computing TF vectors"
        tf = { w:np.zeros(self.ndocs) for w in self.vocab }
        for d, doc in enumerate(all_words):
            # accumulate all word counts
            for w in doc:
                if tf.has_key(w):
                    tf[w][d] += 1.0
        self.tf = tf

        # Get inverse document frequencies
        print "Computing IDF"
        idf = { w:0 for w in self.vocab }
        for w in self.vocab:
            # no. documents / no. of documents which have this word
            idf[w] = self.ndocs * 1.0 / sum(tf[w] != 0)
        self.idf = idf

        # Compute TF-IDF
        print "Computing TF-IDF"
        tfidf = np.zeros((len(self.vocab), self.ndocs))
        for w, key in enumerate(self.vocab):
            logtf = np.ma.log10(tf[key])        # create a masked input to save log10(0)
            tfidf[w] = (1 + logtf).filled(0.)   # fill log(0)'s with 0
            tfidf[w] *= np.log10(idf[key])
        self.tfidf = tfidf


    def match_tfidf_score(self, docid, sentence1, sentence2):
        """
        Compute matching score between 2 sentences
        IMPORTANT: sentence1 and sentence2 are a list of words!
        """
        matching_words = set(sentence1).intersection(sentence2)
        score = 0.
        for w in matching_words:
            if w in self.vocab:
                score += self.tfidf[self.vocab.index(w)][docid]

        return score


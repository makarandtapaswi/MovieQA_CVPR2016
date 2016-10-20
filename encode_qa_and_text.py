#!/usr/bin/python
import os
import sys
import ipdb
import pickle
import word2vec as w2v
import scipy.sparse as sps
sys.path.append('/u/makarand/Codes/UToronto/skip-thoughts')
import skipthoughts
import numpy as np
import progressbar as pb
pb_widgets = ['Encoding: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
# Local imports
import utils
import tfidf as tfidfcalc
# import video-text embedding
# sys.path.append('/ais/guppy9/movie2text/video_text_embed')
# from sentence_encoder import Encoder as VisTextEncoder
from movieqa_importer import MovieQA
mqa = MovieQA.DataLoader()


def check_save_directory(filename=None, dirname=None):
    """Make the folder where descriptors are saved if it doesn't exist.
    """

    if filename:
        dirname = filename.rsplit('/', 1)[0]

    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def encode_tfidf_model(document_type, word_thresh=1):
    """Load TF-IDF model.
    """

    tfidf_fname = utils.TFIDF_TEMPLATE %(document_type, word_thresh)
    check_save_directory(filename=tfidf_fname)

    if os.path.exists(tfidf_fname):
        with open(tfidf_fname, 'rb') as fid:
            TFIDF = pickle.load(fid)

    else:
        # read the story and gather words
        story, _ = mqa.get_story_qa_data('full', document_type)
        sorted_movies = sorted(story.keys())
        all_words_use = []
        for imdb_key in sorted_movies:
            all_words_use.append([])
            for sentence in story[imdb_key]:
                norm_sentence = utils.normalize_stemming(utils.normalize_alphanumeric(sentence.lower()))
                all_words_use[-1].extend(norm_sentence.split(' '))

        # compute TFIDF
        TFIDF = tfidfcalc.TFIDF(sorted_movies)
        TFIDF.get_filtered_vocabulary(all_words_use, word_thresh=word_thresh)
        TFIDF.compute_tfidf(all_words_use)

        # dump to pickle file for future
        with open(tfidf_fname, 'wb') as fid:
            pickle.dump(TFIDF, fid)

    return TFIDF


def load_model(desc, tfidf_doc='split_plot', tfidf_wthr=1):
    """Load appropriate model based on descriptor type.
    """

    model = None
    if desc.startswith('tfidf'):
        model = encode_tfidf_model(tfidf_doc, tfidf_wthr)
        desc = desc + '-' + tfidf_doc + '-' + str(tfidf_wthr)

    elif desc == 'word2vec':
        model = w2v.load('models/movie_plots_1364.d-300.mc1.w2v', kind='bin')

    elif desc == 'skipthought':
        model = skipthoughts.load_model()

    elif desc == 'vis-text-embed':
        raise ValueError('Visual-Text embeddings are not yet supported.')
    #     model = VisTextEncoder()

    return model, desc


def encode_sentences(desc, sentence_list, model, imdb_key=None, is_qa=False):
    """Encode a list of sentences given the model.
    """

    if desc == 'skipthought':
        # encode a sentence list directly
        features = skipthoughts.encode(model, sentence_list, verbose=False)

    elif desc == 'vis-text-embed':
        # normalize sentence lists
        norm_sentence_list = [utils.normalize_alphanumeric(sentence.lower()) for sentence in sentence_list]
        # allows to encode a sentence list directly
        features = model.encode(norm_sentence_list)

    elif desc.startswith('tfidf'):
        desc_dim = len(model.vocab)
        midx = model.doc_names.index(imdb_key)
        # use scipy sparse matrix when encoding stories, otherwise too huge!
        if is_qa:
            features = np.zeros((len(sentence_list), desc_dim), dtype='float32')
        else:
            features = sps.dok_matrix((len(sentence_list), desc_dim), dtype='float32')

        for s, sentence in enumerate(sentence_list):
            # NOTE: use both alphanumeric and stemming normalization
            sentence = utils.normalize_stemming(utils.normalize_alphanumeric(sentence.lower())).split(' ')
            # for each word in the normalized sentence
            for word in sentence:
                if word not in model.vocab: continue
                widx = model.vocab.index(word)
                features[s,widx] = model.tfidf[widx][midx]

            if is_qa:  # if not sparse, use numpy.linalg.norm
                features[s] /= (np.linalg.norm(features[s]) + 1e-6)
            else:  # if sparse, use scipy.sparse.linalg.norm
                features[s] /= (sps.linalg.norm(features[s]) + 1e-6)

    elif desc == 'word2vec':
        desc_dim = model.get_vector(model.vocab[-1]).shape[0]
        features = np.zeros((len(sentence_list), desc_dim), dtype='float32')
        for s, sentence in enumerate(sentence_list):
            # NOTE: use only alphanumeric normalization, no stemming
            sentence = utils.normalize_alphanumeric(sentence.lower()).split(' ')
            # for each word in the normalized sentence
            for word in sentence:
                if word not in model.vocab: continue
                features[s] += model.get_vector(word)

            features[s] /= (np.linalg.norm(features[s]) + 1e-6)

    return features


def encode_documents(document_type, desc, model):
    """Encode sentences from the documents using the descriptor.
    """

    story, _ = mqa.get_story_qa_data('full', document_type)
    check_save_directory(filename=utils.DOC_DESC_TEMPLATE %(document_type, desc, ''))

    pbar = pb.ProgressBar(widgets=pb_widgets, maxval=len(story)).start()
    for i, imdb_key in enumerate(story.keys()):
        pbar.update(i)
        npy_fname = utils.DOC_DESC_TEMPLATE % (document_type, desc, imdb_key)

        # if word2vec file exists continue
        if os.path.exists(npy_fname): continue
        # create a list of sentences
        sentence_list = story[imdb_key]
        # encode sentences
        story_features = encode_sentences(desc, sentence_list, model, imdb_key=imdb_key, is_qa=False)
        # save features, use pickle saver in case of sparse matrix
        if type(story_features) == sps.dok.dok_matrix:
            with open(npy_fname, 'wb') as fid:
                pickle.dump(story_features, fid)
        else:
            np.save(npy_fname, story_features)
    pbar.finish()


def encode_qa(desc, model):
    """Encode question and answer using the descriptor.
    """

    _, QA = mqa.get_story_qa_data('full', 'split_plot')
    check_save_directory(filename=utils.QA_DESC_TEMPLATE %(desc, ''))

    pbar = pb.ProgressBar(widgets=pb_widgets, maxval=len(QA)).start()
    for i, qa in enumerate(QA):
        pbar.update(i)
        npy_fname = utils.QA_DESC_TEMPLATE % (desc, qa.qid)

        # if word2vec file exists continue
        if os.path.exists(npy_fname): continue
        # create a list of sentences
        sentence_list = [qa.question]
        sentence_list.extend([ans for ans in qa.answers if ans])
        # encode sentences, and save features
        qa_features = encode_sentences(desc, sentence_list, model, imdb_key=qa.imdb_key, is_qa=True)
        np.save(npy_fname, qa_features)
    pbar.finish()


def one_pass_encoding(model, desc):
    """Encode all questions and story types using this model.
    """
    ### Encode all questions
    print 'Encoding QA | desc: %s' %(desc)
    encode_qa(desc, model)
    ### Encode all documents
    for doc in reversed(documents):
        print 'Encoding %s | desc: %s' %(doc.upper(), desc)
        encode_documents(doc, desc, model)


if __name__ == '__main__':
    ### Variable types
    documents = ['split_plot', 'script', 'subtitle', 'dvs']
    descriptors = ['tfidf', 'word2vec', 'skipthought'] #, 'vis-text-embed']

    # For each descriptor type
    for desc in descriptors:
        ### Load encoding model
        if desc == 'tfidf':
            model, desc = load_model(desc, tfidf_doc='split_plot', tfidf_wthr=1)
            one_pass_encoding(model, desc)
            model, desc = load_model(desc, tfidf_doc='subtitle', tfidf_wthr=5)
            one_pass_encoding(model, desc)
        else:
            model, desc = load_model(desc)
            one_pass_encoding(model, desc)



#!/usr/bin/python
import sys
import ipdb
import pickle
import numpy as np
import progressbar as pb
pb_widgets = ['Answering: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
from optparse import OptionParser
# Local imports
import utils
from movieqa_importer import MovieQA
mqa = MovieQA.DataLoader()


def moving_average(a, n):
    """Compute the moving average of size 'n' in array 'a'.
    """

    a_sum = np.cumsum(a, axis=0)
    a_summed = a_sum
    a_summed[n:] = a_sum[n:] - a_sum[:-n]
    a_avg = a_summed[n - 1:] / n
    a_avg_norm = np.array([vec / (np.linalg.norm(vec) + 1e-12) for vec in a_avg])
    return np.squeeze(a_avg_norm)


def answer_cosine_similarity(story, QA, story_source, descriptor, window=1):
    """Compute cosine similarity between story, and Q and A features.
    """

    # Use appropriate TFIDF descriptor
    if descriptor == 'tfidf':
        if story_source == 'split_plot':
            descriptor = 'tfidf-split_plot-1'
        else:
            descriptor = 'tfidf-subtitle-5'
        print "Updated descriptor from 'tfidf' to '%s'" %descriptor

    ans_keys = {}
    movies = sorted(story.keys())
    ### Answer questions about one movie at a time, simplify story descriptor loading
    pbar = pb.ProgressBar(widgets=pb_widgets, maxval=len(QA)).start()
    for imdb_key in movies:
        # load story features once
        if descriptor.startswith('tfidf'):
            with open(utils.DOC_DESC_TEMPLATE %(story_source, descriptor, imdb_key), 'rb') as fid:
                story_features = pickle.load(fid)
            story_features = story_features.todense()

        else:
            story_features = np.load(utils.DOC_DESC_TEMPLATE %(story_source, descriptor, imdb_key))

        # window the source text features
        if window > 1:
            story_features = moving_average(story_features, min(story_features.shape[0], window))

        this_QA = [qa for qa in QA if qa.imdb_key == imdb_key]
        for qa in this_QA:
            # load QA features
            qa_features = np.load(utils.QA_DESC_TEMPLATE %(descriptor, qa.qid))
            # compute dot product (cosine similarity ;))
            q_score = np.dot(qa_features[0], story_features.T)
            a_score = np.dot(qa_features[1:], story_features.T)
            # max score across all story-sentences, question, and answers
            a_best_match = [np.max(a + q_score) for a in a_score]
            # add chosen answer
            ans_keys.update({qa.qid: np.argmax(a_best_match)})
            pbar.update(len(ans_keys))
    pbar.finish()
    return ans_keys


def init_option_parser():
    """Initialize parser.
    """

    usage = """
    %prog -s <story_source> [-z <evaluation_set>] [-d <descriptor>] [-w <window>]
    """

    parser = OptionParser(usage=usage)
    parser.add_option("-s", "--story_source", action="store", type="string", default="",
                      help="Story source text: split_plot | dvs | subtitle | script")
    parser.add_option("-d", "--descriptor", action="store", type="string", default="word2vec",
                      help="Descriptor: tfidf | [word2vec] | skipthought")
    parser.add_option("-w", "--window", action="store", type=int, default=1,
                      help="Window over the source text features (default 1)")
    parser.add_option("-z", "--evaluation_set", action="store", type="string", default="val",
                      help="Run final evaluation on? [val] | test")
    return parser


if __name__ == '__main__':
    ### Parse command line options
    parser = init_option_parser()
    opts, args = parser.parse_args(sys.argv)

    assert opts.story_source in ['split_plot', 'dvs', 'subtitle', 'script'], \
        utils.fail_nicely("Invalid story type", parser)
    assert opts.evaluation_set in ['val', 'test'], \
        utils.fail_nicely("Invalid evaluation set", parser)
    assert opts.descriptor in ['tfidf', 'word2vec', 'skipthought'], \
        utils.fail_nicely("Invalid descriptor type", parser)

    ### Load QAs
    # NOTE: As there is no learning involved, training data can be used to
    # select optimal values for the "window" size or other hyperparameters

    # story_train, QA_train = mqa.get_story_qa_data('train', opts.story_source)
    story_eval, QA_eval = mqa.get_story_qa_data(opts.evaluation_set, opts.story_source)

    ### Cosine similarity based answering
    ans_keys = {'cosine.' + opts.story_source + '.' + opts.descriptor + '-w' + str(opts.window): \
                answer_cosine_similarity(story_eval, QA_eval, opts.story_source, opts.descriptor, window=opts.window)}

    ### Evaluate all answer keys
    utils.process_answer_keys(opts.evaluation_set, QA_eval, ans_keys)


#!/usr/bin/python
import os
import sys
import ipdb
import numpy as np
from optparse import OptionParser
from collections import OrderedDict
import scipy.spatial.distance as spdist
import progressbar as pb
pb_widgets = ['Answering: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
# Local imports
import utils
from movieqa_importer import MovieQA
mqa = MovieQA.DataLoader()


def answer_length(QA):
    """Hasty student answering questions based on the length of the answers.
    """

    shortest, longest, different = {}, {}, {}
    for qa in QA:
        # get all answer lengths
        ans_length = np.zeros((5))
        for k, ans in enumerate(qa.answers):
            ans_length[k] = len(utils.normalize_stemming(utils.normalize_alphanumeric(ans)))
        # pick shortest answer
        shortest.update({qa.qid:np.argmin(ans_length)})
        # pick longest answer
        longest.update({qa.qid:np.argmax(ans_length)})
        # pick most different sized answer
        mean_length = np.mean(ans_length)
        different.update({qa.qid:np.argmax(np.abs(ans_length - mean_length))})

    answer_options = {'hasty-shortest': shortest,
                      'hasty-longest': longest,
                      'hasty-different': different}
    return answer_options


def answer_descriptor(QA, desc):
    """Finds the most similar or distinct answer based on all questions
    using descriptors.
    """

    similar, distinct = {}, {}
    pbar = pb.ProgressBar(widgets=pb_widgets, maxval=len(QA)).start()
    for i, qa in enumerate(QA):
        pbar.update(i)
        # load descriptor corresponding to QAs
        feats = np.load(utils.QA_DESC_TEMPLATE %(desc, qa.qid))
        ans_feats = feats[1:]
        # compute cosine distance
        dists = spdist.pdist(ans_feats, metric='cosine')
        dists = np.sum(spdist.squareform(dists), axis=1)
        # pick answers
        similar.update({qa.qid: np.argmin(dists)})
        distinct.update({qa.qid: np.argmax(dists)})
    pbar.finish()

    answer_options = {'hasty-ans-desc-similar-' + desc: similar,
                      'hasty-ans-desc-distinct-' + desc: distinct}
    return answer_options


def question_answer_similarity(QA, desc):
    """Finds the most similar answer by comparing against the question.
    """

    # load descriptor corresponding to QAs
    similar = {}
    pbar = pb.ProgressBar(widgets=pb_widgets, maxval=len(QA)).start()
    for i, qa in enumerate(QA):
        pbar.update(i)
        # load descriptor corresponding to QAs
        feats = np.load(utils.QA_DESC_TEMPLATE %(desc, qa.qid))
        dists = spdist.cdist([feats[0]], feats[1:])
        # pick answer
        similar.update({qa.qid: np.argmin(dists)})
    pbar.finish()

    answer_options = {'hasty-qa-desc-similar-' + desc: similar}
    return answer_options


def init_option_parser():
    """Initialize parser.
    """

    usage = """
    %prog -e <experiment> [-z <evaluation_set>] [-d <descriptor>]
    """

    parser = OptionParser(usage=usage)
    parser.add_option("-e", "--experiment", action="store", type="string", default="",
                      help="Experiment type: answer_length | within_answer | question_answer")
    parser.add_option("-d", "--descriptor", action="store", type="string", default="word2vec",
                      help="Descriptor: tfidf-subtitle-5 | [word2vec] | skipthought")
    parser.add_option("-z", "--evaluation_set", action="store", type="string", default="val",
                      help="Run final evaluation on? [val] | test")
    return parser


if __name__ == '__main__':
    ### Parse command line options
    parser = init_option_parser()
    options, args = parser.parse_args(sys.argv)

    assert options.experiment in ['answer_length', 'within_answer', 'question_answer'], \
        utils.fail_nicely("Invalid experiment type", parser)
    assert options.evaluation_set in ['val', 'test'], \
        utils.fail_nicely("Invalid evaluation set", parser)
    assert options.descriptor in ['tfidf-subtitle-5', 'word2vec', 'skipthought'], \
        utils.fail_nicely("Invalid QA descriptor", parser)

    ### Load QAs
    print "Evaluating on:", options.evaluation_set
    _, QA_eval = mqa.get_story_qa_data(options.evaluation_set, 'split_plot')

    ### Answering
    if options.experiment == 'answer_length':
        ans_keys = answer_length(QA_eval)
    elif options.experiment == 'within_answer':
        ans_keys = answer_descriptor(QA_eval, options.descriptor)
    elif options.experiment == 'question_answer':
        ans_keys = question_answer_similarity(QA_eval, options.descriptor)
    else:
        raise ValueError('Invalid experiment type.')

    ### Process answer keys
    utils.process_answer_keys(options.evaluation_set, QA_eval, ans_keys)


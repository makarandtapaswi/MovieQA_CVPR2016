#!/usr/bin/python
# End-to-End Memory Networks -- Theano implementation
# Modified to run with on the MovieQA dataset

# General imports
import os
import sys
import ipdb
import json
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import time
import word2vec as w2v
from collections import Counter
from optparse import OptionParser
# Local imports
from movieqa_importer import MovieQA
from memN2N_text import MemoryNetwork
import utils


# Seed random number generators
rng = np.random
rng.seed(1234)

w2v_mqa_model_filename = 'models/movie_plots_1364.d-300.mc1.w2v'


def get_minibatch(batch_idx, stM, quesM, ansM, qinfo, mute_targets=False):
    """Create one mini-batch from the data.
    Inputs:
        batch_idx - a vector of indices to select stories, questions from

    Returns:
        memorydata - batchsize X numsentence X numwords - story representation
        inputq - batchsize X numwords - input question representation
        target - batchsize X 1 - correct answer indices during training 0-4
        multians - batchsize X 5 X numwords - multiple choice answer representations
        b_qinfo - batchsize X [] - list of question information
    """

    story_shape = stM.values()[0].shape
    num_ma_opts = ansM.shape[1]

    inputq = np.zeros((len(batch_idx), quesM.shape[1]), dtype='int32')                             # question input vector
    target = np.zeros((len(batch_idx)), dtype='int32')                                             # answer (as a single number)
    memorydata = np.zeros((len(batch_idx), story_shape[0], story_shape[1]), dtype='int32')         # memory statements
    multians = np.zeros((len(batch_idx), num_ma_opts, ansM.shape[2]), dtype='int32')               # multiple choice answers
    b_qinfo = []

    for b, bi in enumerate(batch_idx):
        # question vector
        inputq[b] = quesM[bi]
        # answer option number
        if not mute_targets:
            target[b] = qinfo[bi]['correct_option']
        # multiple choice answers
        multians[b] = ansM[bi]   # get list of answers for this batch
        # get story data
        memorydata[b] = stM[qinfo[bi]['movie']]
        # qinfo
        b_qinfo.append(qinfo[bi])

    return memorydata, inputq, target, multians, b_qinfo


def count_errors(yhat, target):
    """Counts the total number of errors.
    """

    errors = sum(np.argmax(yhat, axis=1) != target)
    return errors


def call_train_epoch(train_func, train_data, train_range, bs=8, lr=0.01, iterprint=False):
    """One epoch of training.
    """

    train_error, train_cost, it = 0., 0., 0

    n_train_batches = int(len(train_range) / bs)
    train_perm = rng.permutation(train_range)
    # iterate over all batches in the data
    for batch_count in xrange(n_train_batches):
        it += 1
        # get indices of this minibatch
        this_batch = train_perm[batch_count * bs : (batch_count + 1) * bs]
        # get minibatch
        memorydata, inputq, target, multians, b_qinfo = \
            get_minibatch(this_batch, train_data['s'], train_data['q'], train_data['a'], train_data['qinfo'])
        # call train model
        cost, yhat, g_norm, p_norm = train_func(memorydata, inputq, target, multians, lr)
        er = count_errors(yhat, target)

        # print iteration info
        if iterprint:
            print "\titer: %5d | train error: %7.3f | batch-cost: %7.3f" %(it, 100.0*er/bs, cost),
            print "| W norms:", p_norm, "| G norms:", g_norm

        # accumulate stuff
        train_cost += cost
        train_error += er
        # train_gnorm += g_norm

    # normalize counts over all batches and samples
    train_error = 100 * train_error / (n_train_batches * bs)
    train_cost = train_cost / n_train_batches

    return train_error, train_cost, it


def call_test(test_func, test_data, data_range=None, bs=8):
    """Run one round of test on all data.
        if data_range is None:
            we are running cleanly on the actual val, or test sets
        else:
            we are running on train-val
    """

    if data_range:  # train-val
        num_qa = len(data_range)
        n_test_batches = int(len(data_range) / bs)
        mute_targets = False
        test_error = 0.
    else:  # val and test
        num_qa = len(test_data['qinfo'])
        n_test_batches = int(np.ceil(1.0*num_qa / bs))
        mute_targets = True
        ans_keys = {}

    # iterate over all batches in the data
    for batch_count in xrange(n_test_batches):
        # get indices of this minibatch
        if data_range:  # train-val
            this_batch = data_range[batch_count * bs : (batch_count + 1) * bs]
        else:  # val and test
            this_batch = range(batch_count * bs, min( (batch_count+1) * bs, len(test_data['qinfo']) ))

        # get minibatch
        memorydata, inputq, target, multians, b_qinfo = \
            get_minibatch(this_batch, test_data['s'], test_data['q'], test_data['a'], test_data['qinfo'], mute_targets=mute_targets)
        # call test function
        yhat = test_func(memorydata, inputq, multians)
        if data_range:  # train-val
            er = count_errors(yhat, target)
            test_error += er
        else:  # val and test
            ans_keys.update({qa['qid']: np.argmax(yhat[k]) for k, qa in enumerate(b_qinfo)})

    if data_range:  # train-val
        test_error = 100 * test_error / (n_test_batches * bs)
        return test_error
    else:  # val and test
        return ans_keys


def create_vocabulary(QAs, stories, v2i, w2v_vocab=None, word_thresh=2):
    """Create the vocabulary by taking all words in stories, questions, and answers taken together.
    Also, keep only words that appear in the word2vec model vocabulary (if provided with one).
    """

    print "Creating vocabulary.",
    if w2v_vocab is not None:
        print "Adding words based on word2vec"
    else:
        print "Adding all words"
    # Get all story words
    all_words = [word for story in stories for sent in story for word in sent]

    # Parse QAs to get actual words
    QA_words = []
    for QA in QAs:
        QA_words.append({})
        QA_words[-1]['q_w'] = utils.normalize_alphanumeric(QA.question.lower()).split(' ')
        QA_words[-1]['a_w'] = [utils.normalize_alphanumeric(answer.lower()).split(' ') for answer in QA.answers]

    # Append question and answer words to all_words
    for QAw in QA_words:
        all_words.extend(QAw['q_w'])
        for answer in QAw['a_w']:
            all_words.extend(answer)

    # threshold vocabulary, at least N instances of every word
    vocab = Counter(all_words)
    vocab = [k for k in vocab.keys() if vocab[k] >= word_thresh]

    # create vocabulary index
    for w in vocab:
        if w not in v2i.keys():
            if w2v_vocab is None:
                # if word2vec is not provided, just dump the word to vocab
                v2i[w] = len(v2i)
            elif w2v_vocab is not None and w in w2v_vocab:
                # check if word in vocab, or else ignore
                v2i[w] = len(v2i)

    print "Created a vocabulary of %d words. Threshold removed %.2f %% words" \
            %(len(v2i), 100*(1. * len(set(all_words)) - len(v2i))/len(all_words))

    return QA_words, v2i


def data_in_matrix_form(stories, QA_words, v2i):
    """Make the QA data set compatible for memory networks by
    converting to matrix format (index into LUT vocabulary).
    """

    def add_word_or_UNK():
        if v2i.has_key(word):
            return v2i[word]
        else:
            return v2i['UNK']

    # Encode stories
    max_sentences = max([len(story) for story in stories.values()])
    max_words = max([len(sent) for story in stories.values() for sent in story])

    storyM = {}
    for imdb_key, story in stories.iteritems():
        storyM[imdb_key] = np.zeros((max_sentences, max_words), dtype='int32')
        for jj, sentence in enumerate(story):
            for kk, word in enumerate(sentence):
                storyM[imdb_key][jj, kk] = add_word_or_UNK()

    print "#stories:", len(storyM)
    print "storyM shape (movie 1):", storyM.values()[0].shape

    # Encode questions
    max_words = max([len(qa['q_w']) for qa in QA_words])
    questionM = np.zeros((len(QA_words), max_words), dtype='int32')
    for ii, qa in enumerate(QA_words):
        for jj, word in enumerate(qa['q_w']):
            questionM[ii, jj] = add_word_or_UNK()
    print "questionM:", questionM.shape

    # Encode answers
    max_answers = max([len(qa['a_w']) for qa in QA_words])
    max_words = max([len(a) for qa in QA_words for a in qa['a_w']])
    answerM = np.zeros((len(QA_words), max_answers, max_words), dtype='int32')
    for ii, qa in enumerate(QA_words):
        for jj, answer in enumerate(qa['a_w']):
            if answer == ['']:  # if answer is empty, add an 'UNK', since every answer option should have at least one valid word
                answerM[ii, jj, 0] = 1
                continue
            for kk, word in enumerate(answer):
                answerM[ii, jj, kk] = add_word_or_UNK()
    print "answerM:", answerM.shape

    return storyM, questionM, answerM


def associate_additional_QA_info(QAs):
    """Get some information about the questions like story index and correct option.
    """

    qinfo = []
    for QA in QAs:
        qinfo.append({'qid':QA.qid,
                      'movie':QA.imdb_key,
                      'correct_option':QA.correct_index})
    return qinfo


def normalize_documents(stories, normalize_for=('lower', 'alphanumeric'), max_words=40):
    """Normalize all stories in the dictionary, get list of words per sentence.
    """

    for movie in stories.keys():
        for s, sentence in enumerate(stories[movie]):
            sentence = sentence.lower()
            if 'alphanumeric' in normalize_for:
                sentence = utils.normalize_alphanumeric(sentence)
            sentence = sentence.split(' ')[:max_words]
            stories[movie][s] = sentence
    return stories


def main(options):
    """Main function which wraps everything.
        - Prepare data: word2vec, vocabulary creation, train/val/test splits
        - Build MemoryNetwork Theano model
        - Run training pass
    """

    print "----------- Prepare data ------------"
    # Get list of MAs and movies
    mqa = MovieQA.DataLoader()

    ### Process story source
    stories, QAs = mqa.get_story_qa_data('full', options['data']['source'])
    stories = normalize_documents(stories)

    ### Load Word2Vec model
    w2v_model = w2v.load(w2v_mqa_model_filename, kind='bin')
    options['memnn']['w2v'] = w2v_model
    options['memnn']['d-w2v'] = len(w2v_model.get_vector(w2v_model.vocab[1]))
    print "Loaded word2vec model: dim = %d | vocab-size = %d" \
        %(options['memnn']['d-w2v'], len(w2v_model.vocab))

    ### Create vocabulary-to-index and index-to-vocabulary
    v2i = {'': 0, 'UNK':1}  # vocabulary to index
    QA_words, v2i = create_vocabulary(QAs, stories, v2i,
                                 w2v_vocab=w2v_model.vocab.tolist(),
                                 word_thresh=options['data']['vocab_threshold'])
    i2v = {v:k for k,v in v2i.iteritems()}

    ### Convert QAs and stories into numpy matrices (like in the bAbI data set)
    # storyM - Dictionary - indexed by imdb_key. Values are [num-sentence X max-num-words]
    # questionM - NP array - [num-question X max-num-words]
    # answerM - NP array - [num-question X num-answer-options X max-num-words]
    storyM, questionM, answerM = data_in_matrix_form(stories, QA_words, v2i)
    qinfo = associate_additional_QA_info(QAs)

    ### Split everything into train, val, and test data
    train_storyM = {k:v for k, v in storyM.iteritems() if k in mqa.data_split['train']}
    val_storyM   = {k:v for k, v in storyM.iteritems() if k in mqa.data_split['val']}
    test_storyM  = {k:v for k, v in storyM.iteritems() if k in mqa.data_split['test']}

    def split_train_test(long_list, QAs, trnkey='train', tstkey='val'):
        # Create train/val/test splits based on key
        train_split = [item for k, item in enumerate(long_list) if QAs[k].qid.startswith('train')]
        val_split = [item for k, item in enumerate(long_list) if QAs[k].qid.startswith('val')]
        test_split = [item for k, item in enumerate(long_list) if QAs[k].qid.startswith('test')]
        if type(long_list) == np.ndarray:
            return np.array(train_split), np.array(val_split), np.array(test_split)
        else:
            return train_split, val_split, test_split

    train_questionM, val_questionM, test_questionM = split_train_test(questionM, QAs)
    train_answerM,   val_answerM,   test_answerM,  = split_train_test(answerM, QAs)
    train_qinfo,     val_qinfo,     test_qinfo     = split_train_test(qinfo, QAs)

    QA_train = [qa for qa in QAs if qa.qid.startswith('train:')]
    QA_val   = [qa for qa in QAs if qa.qid.startswith('val:')]
    QA_test  = [qa for qa in QAs if qa.qid.startswith('test:')]

    train_data = {'s':train_storyM, 'q':train_questionM, 'a':train_answerM, 'qinfo':train_qinfo}
    val_data =   {'s':val_storyM,   'q':val_questionM,   'a':val_answerM,   'qinfo':val_qinfo}
    test_data  = {'s':test_storyM,  'q':test_questionM,  'a':test_answerM,  'qinfo':test_qinfo}

    ### Build model
    print "------------ Build model ------------"
    memnn = MemoryNetwork(options['memnn'], rng)
    memnn.setup_model_configuration(v2i, storyM.values()[0].shape)
    memnn.build_model()
    memnn.gradients_and_updates(grad_normalize=options['train']['gnorm'])

    ### Get model functions
    tic = time.time()
    train_func = memnn.train_function()
    test_func = memnn.test_function()
    print "Model building completed in %.2f(s)" %(time.time() - tic)

    ### Main Training loop
    print "-------------- Training -------------"

    # Make train/val splits within train set based on movie
    with open('train_split.json') as fid:
        trdev = json.load(fid)
    train_range = [k for k, qi in enumerate(qinfo) if qi['movie'] in trdev['train']]
    val_range   = [k for k, qi in enumerate(qinfo) if qi['movie'] in trdev['dev']]

    # Start main epoch loop
    ans_string = 'memnn_text.%s.ep-%s.nlayer-%d.d_lproj-%d.lr-%.2f' \
            %(options['data']['source'], '%03d', options['memnn']['num_mem_layers'],
              options['memnn']['d_lproj'], options['train']['learning_rate'])
    ep, it = 0, 0
    val_keys, test_keys = {}, {}
    min_train_val_error = {'error': 100, 'epoch': -1}  # least train-val error during the run @ '-1' epoch
    while ep < options['train']['nepochs']:
        tic = time.time()
        # Training pass
        tr_e, tr_c, it_e = call_train_epoch(train_func, train_data, train_range,
                                bs=options['train']['batch_size'], lr=options['train']['learning_rate'], iterprint=False)
        ep += 1
        it += it_e
        print "epoch: %3d | iter: %4d | train error: %.3f | batch-cost: %.3f | time: %.3f(s)" \
                        % (ep, it, tr_e, tr_c, time.time()-tic)

        # Train-val pass
        train_val_error = call_test(test_func, train_data, data_range=val_range, bs=options['train']['batch_size'])
        print "epoch: %3d | train-val error: %.3f" %(ep, train_val_error)

        # Decide to run on test-set if train-val error is lower than current minimum
        if train_val_error < min_train_val_error['error']:
            min_train_val_error['error'] = train_val_error
            min_train_val_error['epoch'] = ep

            # Validation and Test pass
            print "Encountered minimum train-val error. Running on Val and Test"
            val_keys =  {ans_string %(ep): call_test(test_func, val_data,  bs=options['train']['batch_size'])}
            test_keys = {ans_string %(ep): call_test(test_func, test_data, bs=options['train']['batch_size'])}

        # Reduce learning rate?
        if np.mod(ep, options['train']['lrdecay'][1]) == 0:
            print "Reducing learning rate from", options['train']['learning_rate'],
            options['train']['learning_rate'] *= options['train']['lrdecay'][0]
            print "to", options['train']['learning_rate']

    # Push best epoch answer keys for evaluation
    best_ep = min_train_val_error['epoch']
    eval_val =  {ans_string %(best_ep):  val_keys[ans_string %(best_ep)]}
    eval_test = {ans_string %(best_ep): test_keys[ans_string %(best_ep)]}
    utils.process_answer_keys('val',  QA_val,  eval_val)
    utils.process_answer_keys('test', QA_test, eval_test)

    # call_test(test_func, val_data, bs=options['train']['batch_size'])
    # call_test(test_func, test_data, bs=options['train']['batch_size'])


def init_option_parser():
    """Initialize parser.
    """

    usage = """
    Important options are printed here. Check out the code more tweaks.
    %prog -s <story_source> [-n <num_mem_layers>]
                    [--learning_rate <lr>] [--batch_size <bs>] [--nepochs <ep>]
    """

    parser = OptionParser(usage=usage)
    parser.add_option("-s", "--story_source", action="store", type="string", default="",
                      help="Story source text: split_plot | dvs | subtitle | script")
    parser.add_option("-n", "--num_mem_layers", action="store", type=int, default=1,
                      help="Number of Memory layers")
    parser.add_option("",   "--batch_size", action="store", type=int, default=8,
                      help="Batch size. Ranges from 8 to 64 (depends on GPU memory)")
    parser.add_option("",   "--learning_rate", action="store", type=float, default=0.01,
                      help="Initial learning rate for SGD")
    parser.add_option("",   "--nepochs", action="store", type=int, default=100,
                      help="Train for N epochs")
    return parser


if __name__ == '__main__':
    ### Parse command line options
    parser = init_option_parser()
    opts, args = parser.parse_args(sys.argv)

    assert opts.story_source in ['split_plot', 'dvs', 'subtitle', 'script'], \
        utils.fail_nicely("Invalid story type", parser)
    print 'Evaluating Memory Networks (Text) on MovieQA using: %s' % opts.story_source

    # -------------------------------------------------------
    # Initialize options, lots of defaults, some from parser
    # -------------------------------------------------------
    options = {'memnn':{}, 'train':{}, 'data':{}}
    # MemN2N options
    options['memnn']['num_mem_layers'] = opts.num_mem_layers    # number of memory layers
    options['memnn']['embed_dimension'] = 300                   # learn LUT -- word embedding dimension
    options['memnn']['d_lproj'] = 300                           # dimension for linear projection (100, 300)
    # Training options
    options['train']['nepochs'] = opts.nepochs                  # number of train epochs
    options['train']['batch_size'] = opts.batch_size            # batch size
    options['train']['learning_rate'] = opts.learning_rate      # learning rate
    options['train']['lrdecay'] = [0.9, 10]                     # every [10] epochs, lr = lr * [0.9]
    options['train']['validate_after'] = 1                      # number of epochs to run validation after
    options['train']['gnorm'] = {'max_norm': 40}                # gradient normalization options 'max_norm' OR 'clip'
    # Data options
    options['data']['source'] = opts.story_source               # use this data source for answering questions
    options['data']['learn_LUT'] = False                        # learn / load LUTs -- depending on data source
    options['data']['vocab_threshold'] = 1                      # word must occur >= N times for it to be part of vocabulary

    ### Deprecated - keep code simpler!
    # options['mode'] = 'multi_choice'                          # QA mode, 'single' vs. 'multi_choice'
    # options['memnn']['position_encode'] = False               # encode position by weighting words in a sentence
    # options['memnn']['temporal_encode'] = False               # encode time by adding extra "word" at end of sentence
    # options['memnn']['randomize_time'] = 0.1                  # make the time encoding a bit noisy
    # options['memnn']['l2_regularize'] = False                 # L2 regularization on the parameters
    # options['memnn']['weight_sharing'] = "all"                # share weights between layers? types: 'adjacent', 'rnn', 'all'
    # options['memnn']['replace_LUT_with_w2v'] = True           # THIS IS THE CASE! replace LUT with word2vec initialized vectors, use extra embedding
    # options['memnn']['init_LUT'] = 'randn'                    # weight initialization method for LUT based embeddings: 'randn', 'w2v'

    # -------------------------------------------------------

    # Go, go, go!
    main(options)
    sys.exit()

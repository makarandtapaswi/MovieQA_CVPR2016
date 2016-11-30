#!/usr/bin/python
import os
import sys
import ipdb
import json
import time
import collections
import numpy as np
import cPickle as pkl
import progressbar as pb
from optparse import OptionParser
sys.path.insert(0, '/h/14/makarand/.local/lib/python2.7/site-packages/')
sys.path.insert(0, '/pkgs/theano-12Oct2016/')
sys.path.append('/pkgs/theano-0.8/')
import theano
import theano.tensor as tensor
import lasagne
print "Theano:", theano.__version__, "& Lasagne:", lasagne.__version__

# Local imports
import utils
from movieqa_importer import MovieQA


class MaxAlongLayer(lasagne.layers.Layer):
    """Compute Max along a specified axis of layer.
    """

    def __init__(self, incoming, axis=0, **kwargs):
        super(MaxAlongLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_for(self, input, **kwargs):
        pool = tensor.max(input, axis=self.axis)
        return pool


class AvgAlongLayer(lasagne.layers.Layer):
    """Compute Average along a specified axis of layer.
    """

    def __init__(self, incoming, axis=0, **kwargs):
        super(AvgAlongLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_for(self, input, **kwargs):
        pool = tensor.mean(input, axis=self.axis)
        return pool


class SoftmaxLayer(lasagne.layers.Layer):
    """Compute Softmax.
    """

    def get_output_for(self, input, **kwargs):
        input = input + 1e-10
        return tensor.nnet.softmax(input)


def build_baseline_net(opts):
    """Builds a baseline network to simulate searching student.
    The idea of similarity <s, q> and <s, a> is used in both methods.
    First convolutional layer of SSCB learns weights, while searching student uses equal weight.
    """

    # Input shape: batch_size, column, row, depth,
    inputs = tensor.ftensor4('inputs')
    # Shuffle input dimension -> [batch_size, depth, row, column]
    shuffled = inputs.dimshuffle((0, 3, 2, 1))
    # Filter shape (# filters, depth, row, col)
    filters = tensor.as_tensor_variable(np.ones((1, 2 * len(opts.descriptor), 1, 1)).astype(np.float32))
    # Conv: batch * 1 * row * col
    conv = tensor.nnet.conv.conv2d(shuffled, filters)
    # Pool: batch * 1 * row * 1
    pool = tensor.max(conv, axis=3)
    pool = pool.dimshuffle((0, 2))
    pred_label = tensor.argmax(pool, axis=1)
    return inputs, pred_label


class SSCB(object):
    """Searching Student with Convolutional Brain.
    Implements the method for using convolutions to compute similarity
    across story sentences, questions and answers.
    """

    def __init__(self, use_story_len, opts):
        """Training hyperparameters set here.
        """

        self.learning_rate = opts.learning_rate
        self.weight_decay = 0.01  # regularization multiplier
        self.batch_size = opts.batch_size
        self.num_epochs = opts.nepochs
        self.use_story_len = use_story_len
        self.prng = np.random.RandomState(1234)  # random shuffling

    def _build_network(self):
        """Main function to create the network architecture.
        """

        self.inputs = tensor.ftensor4('inputs')
        self.targets = tensor.ivector('targets')

        # shape = (batch_size, depth = [q, a] * num-descriptors, row = story sentences, columns = 5)
        network = lasagne.layers.InputLayer(
            shape=(None, self.use_story_len, 5, 2 * len(opts.descriptor)),
            input_var=self.inputs)
        network = lasagne.layers.DimshuffleLayer(network, (0, 3, 1, 2))

        # input shape = (batch_size, 2, max_plot_len, 5 answers)
        network = lasagne.layers.Conv2DLayer(network, num_filters=10,
            filter_size=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            b=lasagne.init.Constant(0.01),
            W=lasagne.init.HeUniform())

        network = lasagne.layers.Conv2DLayer(network, num_filters=10,
            filter_size=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            b=lasagne.init.Constant(0.01),
            W=lasagne.init.HeUniform())

        # Max pooling here partially resembles the windowing in Searching-Student
        network = lasagne.layers.MaxPool2DLayer(network, (3, 1))

        # Another conv after pool
        network = lasagne.layers.Conv2DLayer(network, num_filters=1,
            filter_size=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            b=lasagne.init.Constant(0.01),
            W=lasagne.init.HeUniform())

        # Do a weighted sum of the average and max across story sentences.
        # 0.3 * avg + 1 * max
        network_b1 = AvgAlongLayer(network, axis=2)
        network_b2 = MaxAlongLayer(network, axis=2)
        network = lasagne.layers.ElemwiseSumLayer([network_b1, network_b2], [0.3, 1])
        network = lasagne.layers.ReshapeLayer(network, [-1, 5])

        # Normalize softmax output layer (batch_size x 5)
        self.network = SoftmaxLayer(network)

    def _build_ce_loss(self):
        """Cross-entropy loss for answer classification.
        Also builds gradients, scales them, and performs Adam updates.
        """

        # Get outputs and compute CE loss
        train_prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.categorical_crossentropy(train_prediction, self.targets)
        loss = loss.mean()

        # Get parameters, and add regularization factor to loss
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        reg_cost = lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
        loss += self.weight_decay * reg_cost

        # Compute gradients for paramters, scale and normalize
        all_grads = tensor.grad(loss, params)
        scaled_grads, grad_norm = lasagne.updates.total_norm_constraint(all_grads, 10, return_norm=True)
        updates = lasagne.updates.adam(scaled_grads, params, learning_rate=self.learning_rate)

        # Test-time computation
        test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                self.targets)
        test_loss = test_loss.mean()
        test_pred_label = tensor.argmax(test_prediction, axis=1)
        test_acc = tensor.mean(tensor.eq(test_pred_label, self.targets),
                               dtype=theano.config.floatX)

        # Compiling the model
        print 'Compiling Theano train and test functions... ',
        self.train_fn = theano.function([self.inputs, self.targets], [loss, reg_cost, grad_norm], updates=updates)
        self.test_fn = theano.function([self.inputs, self.targets], [test_loss, test_acc, test_prediction])
        # Eval function used to test when labels are not available. Cannot compute test accuracy
        self.eval_fn = theano.function([self.inputs], test_pred_label)
        print 'done.'

    def _get_batch(self, np_inputs, np_targets, shuffle=False):
        """Create batches of data.
        """

        if len(np_inputs) != len(np_targets):
            raise ValueError('Unmatched inputs and targets size!')

        np_inputs = np_inputs.astype(np.float32)
        np_targets = np_targets.astype(np.int32)
        # Create a shuffled order of indices
        if shuffle:
            indices = np.arange(len(np_inputs))
            self.prng.shuffle(indices)
        # Yield a batch of data
        batch_size = self.batch_size
        for start_idx in range(0, len(np_inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx : start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield np_inputs[excerpt], np_targets[excerpt]

    def run_all(self, opts, trn_info, dev_info, val_info, tst_info):
        """Call main training and testing functions in the loop.
        """

        # Decompose info. VAL and TST do not have labels. Return qids and predictions
        trn_data, trn_label = trn_info
        dev_data, dev_label = dev_info
        val_data, val_qids = val_info
        tst_data, tst_qids = tst_info

        # Build the network and setup train and test functions
        self._build_network()
        self._build_ce_loss()

        dev_record = []
        ans_string = 'sscb.' + opts.story_source + '.desc-' + '+'.join(opts.descriptor) + '.ep-%d.lr-' + '%f'%opts.learning_rate
        min_dev_error = {'error': 100, 'epoch': -1, 'accuracy': 0}  # least train-val (dev) error during the run @ '-1' epoch
        for epoch in range(self.num_epochs):
            trn_err, trn_batches = 0, 0
            start_time = time.time()
            # Compute things for a batch of data
            for batch in self._get_batch(trn_data, trn_label, shuffle=True):
                inputs, targets = batch
                out = self.train_fn(inputs, targets)
                err = out[0]
                weight_norm = out[1]
                grad_norm = out[2:]
                print 'err {:.6f} weight_norm {:.6f}, grad_norm {:.6f}\r'.format(1.0 * err, 1.0 * weight_norm, 1.0 * np.sum(grad_norm)),
                sys.stdout.flush()
                trn_err += err
                trn_batches += 1

            # Make a full pass over the train-val (dev) data:
            dev_err, dev_acc, dev_pred = self.test_fn(dev_data, dev_label)

            # Print results for this epoch:
            print '\nEpoch {} of {} took {:.3f}s'.format(
                epoch + 1, self.num_epochs, time.time() - start_time)
            print '\ttraining  loss:\t\t{:.6f}'.format(trn_err / trn_batches)
            print '\ttrain-dev loss:\t\t{:.6f}'.format(1.0 * dev_err)
            print '\ttrain-dev  acc:\t\t{:.2f}%'.format(dev_acc * 100)

            # Run on VAL and TEST if dev error is lower than current minimum
            if dev_err < min_dev_error['error']:
                min_dev_error = {'error': dev_err, 'epoch': epoch, 'accuracy': dev_acc}

                # Validation and Test pass
                print "Encountered minimum train-val (dev) error. Running on Val and Test"
                val_pred = self.eval_fn(val_data)
                tst_pred = self.eval_fn(tst_data)

                # Create answer keys to submit to evaluation
                val_keys  = {ans_string %(epoch): {val_qids[k]:val_pred[k] for k in range(len(val_qids))}}
                tst_keys  = {ans_string %(epoch): {tst_qids[k]:tst_pred[k] for k in range(len(tst_qids))}}

        # end epoch loop

        # Return the best answer keys for val and test and evaluate
        return val_keys, tst_keys


def build_data_and_label(storyqa_first_layer, data_set, this_is_test=False):
    """Creates a input data matrix (first layer of SSCB) of shape (N x 5 x 2) and a label (target) vector.
    When operating with test sets (this_is_test=True), returns qids instead.
    """

    data = []
    label = []  # A vector of len(data)
    qids = []
    for k in data_set:
        ex = storyqa_first_layer[k]
        for qa_id in range(len(ex['ans'])):
            # num_story, 5, 2 * len(feats)
            data_feat = []
            for feat in opts.descriptor:
                data_feat.append(np.asarray(ex[feat]['sq_feat'][qa_id].T[..., np.newaxis])) # num_story, 5
                data_feat.append(np.asarray(ex[feat]['sa_feat'][qa_id].T[..., np.newaxis]))
            # print len(data_feat), data_feat[0].shape, data_feat[1].shape
            data.append(np.concatenate(data_feat, axis=2))
            label.append(ex['ans'][qa_id][0])
            qids.append(ex['ans'][qa_id][1])
    if this_is_test:
        return data, qids
    else:
        return data, np.asarray(label).astype(np.int32)


def padding_input_data(storyqa_first_layer, target_len=None, num_ans=5, pad_percentile=100):
    """Zero pad (or truncate) first layer of SSCB data based on story length.
    """

    # shapes is a list of (N x 5 x 2) corresponding to each question.
    shapes = [ex.shape for ex in storyqa_first_layer]
    shapes = np.asarray(shapes)
    assert np.all(shapes[:, 1] == num_ans)  # check that every thing has 5 answers
    assert np.all(shapes[:, 2] == shapes[0, 2])  # all tensors should have same depth (2 * len(descriptors))

    # pick target length of padding as <some> percentile of the all story sizes
    target_len = target_len if target_len is not None else int(np.percentile(shapes[:, 0], pad_percentile))
    # 100 percentile means max
    # something like 95 saves a ton of memory by ignoring a few outliers

    if np.all(shapes[:, 0] == shapes[0, 0]):
        pass
    else:
        for i in range(len(storyqa_first_layer)):
            if len(storyqa_first_layer[i]) < target_len:  # Do zero padding
                storyqa_first_layer[i] = np.pad(storyqa_first_layer[i],
                    [(0, target_len - storyqa_first_layer[i].shape[0]), (0, 0), (0, 0)],
                    'constant', constant_values=0)
            else:  # Truncate to desired length
                storyqa_first_layer[i] = storyqa_first_layer[i][:target_len]

    # Return padded/truncated versions of the story
    return np.asarray(storyqa_first_layer).astype(np.float32), target_len


def load_text_features(opts, all_movies, all_QA, num_ans=5):
    """Loads text features, and prepares the first tensor of the SSCB
        - <question, story> feature dot products replicated 5 times.
        - <answer, story> feature dot products provide 5 numbers already.
    """

    pb_widgets = ['Loading QA features: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    qa_features = collections.defaultdict(dict)
    pbar = pb.ProgressBar(widgets=pb_widgets, maxval=len(all_QA)).start()
    for k, qa in enumerate(all_QA):
        for feat in opts.descriptor:
            if feat not in qa_features[qa.imdb_key].keys():
                qa_features[qa.imdb_key].update({feat: collections.defaultdict(dict),
                                                 'ans': collections.defaultdict(dict)})
            qa_features[qa.imdb_key][feat][qa.qid] = utils.load_qa_feature(qa, feat)
            qa_features[qa.imdb_key]['ans'][qa.qid] = (qa.correct_index, qa.qid)
        pbar.update(k)
    pbar.finish()

    pb_widgets = ['Loading Story features, computing <s,q> and <s,a>: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    layer1 = collections.defaultdict(dict)
    pbar = pb.ProgressBar(widgets=pb_widgets, maxval=len(all_movies)).start()
    for i, key in enumerate(all_movies):
        for feat in opts.descriptor:
            layer1[key]['ans'] = []
            layer1[key][feat] = {'sq_feat': [], 'sa_feat': []}
            # Load story features
            story_feat = utils.load_story_feature(key, opts.story_source, feat)
            # Compute dot product with QA features
            for qid in qa_features[key][feat].keys():
                qa_feat = utils.load_qa_feature(qa=None, feature=feat, qid=qid)
                sq_feat = np.dot(qa_feat[0], story_feat.T)
                layer1[key][feat]['sq_feat'].append(np.tile(sq_feat.reshape((1, -1)), [num_ans, 1]))
                sa_feat = np.dot(qa_feat[1:], story_feat.T)
                # make sure this thing has 5 dim
                if sa_feat.shape[0] < num_ans:
                    sa_feat = np.vstack((sa_feat, np.zeros((num_ans - sa_feat.shape[0], sa_feat.shape[1]))))
                layer1[key][feat]['sa_feat'].append(sa_feat)
                layer1[key]['ans'].append(qa_features[key]['ans'][qid])

                if feat.startswith('tfidf'):
                    layer1[key][feat]['sq_feat'][-1] = np.sqrt(layer1[key][feat]['sq_feat'][-1])
                    layer1[key][feat]['sa_feat'][-1] = np.sqrt(layer1[key][feat]['sa_feat'][-1])
        pbar.update(i)
    pbar.finish()
    return layer1


def init_option_parser():
    """Initialize parser.
    """

    usage = """
    Important options are printed here. Check out the code more tweaks.
    %prog -s <story_source> -d <descriptor>
         [--learning_rate <lr>] [--batch_size <bs>] [--nepochs <ep>]
    """

    parser = OptionParser(usage=usage)
    parser.add_option("-s", "--story_source", action="store", type="string", default="",
                      help="Story source text: split_plot | dvs | subtitle | script")
    parser.add_option("-d", "--descriptor", action="store", type="string", default="",
                      help="Descriptor: tfidf | word2vec | skipthought | <combinations with plus, e.g. tfidf+word2vec+skipthought>")
    parser.add_option("",   "--batch_size", action="store", type=int, default=500,
                      help="Batch size. Ranges from 8 to 64 (depends on GPU memory)")
    parser.add_option("",   "--learning_rate", action="store", type=float, default=0.003,
                      help="Initial learning rate for SGD")
    parser.add_option("",   "--nepochs", action="store", type=int, default=100,
                      help="Train for N epochs")
    parser.add_option("",   "--pad_percentile", action="store", type=int, default=100,
                      help="Stories have different number of sentences. This number determines to which percentile\
                            other stories should be padded. For example, 100 (default) indicates max length. 50 indicates median.")
    return parser


def normalize_zeromean_unitstd(data, mean=None, std=None):
    """Normalize data when given mean/std. Otherwise compute and normalize.
    """

    if mean is None or std is None:
        all_data = np.vstack(data)  # data is a list of 3D tensors of N_i x 5 x 2
        mean = all_data.mean()
        std = all_data.std()

    data = [(d - mean)/std for d in data]
    return data, mean, std


if __name__ == '__main__':
    ### Parse command line options
    parser = init_option_parser()
    opts, args = parser.parse_args(sys.argv)

    assert opts.story_source in ['split_plot', 'dvs', 'subtitle', 'script'], \
        utils.fail_nicely("Invalid story type", parser)
    opts.descriptor = opts.descriptor.split('+')

    # Loader of stories, QAs
    mqa = MovieQA.DataLoader()

    # Pick the list of valid movies and QA based on story source
    usable_stories, usable_QA = mqa.get_story_qa_data(split='full', story_type=opts.story_source)
    usable_movies = usable_stories.keys()

    ### Cache features, makes it easier to load again and again ;)
    # The feature contains the first layer of the network. N x 5 x 2 * len(opts.descriptor)
    cache_filename = 'sscb_cache/%s_%s.pkl' % (opts.story_source, opts.descriptor)
    try:
        with open(cache_filename, 'r') as f:
            storyqa_first_layer = pkl.load(f)
        print "Loaded features from cache!"
    except:
        print "Caching features to pickle file."
        storyqa_first_layer = load_text_features(opts, usable_movies, usable_QA)
        with open(cache_filename, 'w') as f:
            pkl.dump(storyqa_first_layer, f)

    # Split train set into train and dev.
    with open('train_split.json', 'r') as fid:
        trdev_split = json.load(fid)

    # Get a list of train, dev, val, test movies
    trn_movies = [m for m in trdev_split['train']   if m in usable_movies]
    dev_movies = [m for m in trdev_split['dev']     if m in usable_movies]
    val_movies = [m for m in mqa.data_split['val']  if m in usable_movies]
    tst_movies = [m for m in mqa.data_split['test'] if m in usable_movies]

    # Generates data and labels
    # train_data: numpy array [num_example, num_story, 5, 2]
    trn_data, trn_label = build_data_and_label(storyqa_first_layer, trn_movies)
    dev_data, dev_label = build_data_and_label(storyqa_first_layer, dev_movies)
    val_data, val_qids  = build_data_and_label(storyqa_first_layer, val_movies, this_is_test=True)
    tst_data, tst_qids  = build_data_and_label(storyqa_first_layer, tst_movies, this_is_test=True)

    # Normalize mean and standard deviation (before padding!)
    trn_data, data_mean, data_std = normalize_zeromean_unitstd(trn_data)
    dev_data, _, _ = normalize_zeromean_unitstd(dev_data, mean=data_mean, std=data_std)
    val_data, _, _ = normalize_zeromean_unitstd(val_data, mean=data_mean, std=data_std)
    tst_data, _, _ = normalize_zeromean_unitstd(tst_data, mean=data_mean, std=data_std)

    # Pad story lengths with zeros to create batches easily
    trn_data, stlen = padding_input_data(trn_data, pad_percentile=opts.pad_percentile)
    dev_data, _     = padding_input_data(dev_data, stlen, pad_percentile=opts.pad_percentile)
    val_data, _     = padding_input_data(val_data, stlen, pad_percentile=opts.pad_percentile)
    tst_data, _     = padding_input_data(tst_data, stlen, pad_percentile=opts.pad_percentile)
    print "TRN:", trn_data.shape
    print "DEV:", dev_data.shape
    print "VAL:", val_data.shape
    print "TST:", tst_data.shape
    print "Story length set to:", stlen

    # -------------- TEST SSCB NET --------------------
    sscb = SSCB(use_story_len=stlen, opts=opts)
    eval_val, eval_tst = sscb.run_all(opts, [trn_data, trn_label], [dev_data, dev_label], [val_data, val_qids], [tst_data, tst_qids])

    QA_val = [qa for qa in usable_QA if qa.qid.startswith('val')]
    QA_tst = [qa for qa in usable_QA if qa.qid.startswith('test')]

    print "--- VAL ---"
    utils.process_answer_keys('val',  QA_val, eval_val)
    print "--- TEST ---"
    utils.process_answer_keys('test', QA_tst, eval_tst)


"""End-to-End Memory Networks (http://arxiv.org/abs/1503.08895)
Modified heavily to work with MovieQA dataset.
- Can take multiple answer options.
- Are not end-to-end! Cannot learn embeddings for 13k+ words!
- Completely shared linear projection layer.
"""

# General imports
import numpy as np
from collections import OrderedDict
from sklearn.decomposition import PCA
# Theano imports
import theano
import theano.tensor as T
from theano.ifelse import ifelse as ifel
#from theano.compile.nanguardmode import NanGuardMode

theano.config.floatX = 'float32'
theano.config.exception_verbosity = 'high'



def init_linear_projection(rng, nrows, ncols, pca_mat=None):
    """ Linear projection (for example when using fixed w2v as LUT """
    if nrows == ncols:
        P = np.eye(nrows)
        print "Linear projection: initialized as identity matrix"
    else:
        assert([nrows, ncols] == pca_mat.shape, 'PCA matrix not of same size as RxC')
        P = 0.1 * pca_mat
        print "Linear projection: initialized with 0.1 PCA"

    return P.astype(theano.config.floatX)


class MemoryLayer(object):
    """
    Single layer of the memory network.
    """

    def __init__(self, layer_opts):
        """Setup layer options.
        """
        self.T_w2v = layer_opts['T_w2v']

    def make_layer(self, n_params, T_u, T_story, T_mask, rng):
        """
        Inputs:
                network params      (n_params)
                question vector     (T_u)
                story tensor        (T_story)
        Outputs: output vector      (T_o)
        """

        # ------ Encode encoder story data
        T_w2v_out = self.T_w2v[T_story] * T_mask[T_story]
        T_m = T.sum(T_w2v_out, axis=2)
        T_m_norm = T.sqrt(T.sum(T_m ** 2, axis=2))
        T_m = T_m / (T_m_norm.dimshuffle(0, 1, 'x') + 1e-6)
        T_m = T.dot(T_m, n_params['T_B'])

        # ------ Encode decoder story data
        T_w2v_out = self.T_w2v[T_story] * T_mask[T_story]
        T_c = T.sum(T_w2v_out, axis=2)
        T_c_norm = T.sqrt(T.sum(T_c ** 2, axis=2))
        T_c = T_c / (T_c_norm.dimshuffle(0, 1, 'x') + 1e-6)
        T_c = T.dot(T_c, n_params['T_B'])

        # ------ Sentence picker: tensor3-matrix product
        T_p = T.nnet.softmax(T.batched_dot(T_m, T_u))

        # ------ Sum over story decoder
        T_p_2 = T_p.dimshuffle(0, 1, 'x')
        T_o = T.sum(T_p_2 * T_c, axis=1)

        # Collect
        return T_o, T_p


class MemoryNetwork(object):
    """End-to-End Memory Network (modified for MovieQA).
    """

    def __init__(self, opts, rng):
        """Copy options.
        """

        self.rng     = rng
        self.nl      = opts['num_mem_layers']
        self.d_lproj = opts['d_lproj']
        self.w2v     = opts['w2v']
        self.d_w2v   = opts['d-w2v']

        print "Initializing MemoryNetwork - Text"
        print "Using Word2Vec for Look-Up-Table, and a linear projection layer."
        print "#Memory Layers:", self.nl
        print "d-Linear proj:", self.d_lproj

    def setup_model_configuration(self, v2i, story_shape):
        """Setup some configuration parts of the model.
        """

        self.v2i = v2i
        self.vs = len(v2i)

        # define Look-Up-Table mask
        np_mask = np.vstack((np.zeros(self.d_w2v), np.ones((self.vs - 1, self.d_w2v))))
        self.T_mask = theano.shared(np_mask.astype(theano.config.floatX), name='LUT_mask')

        # setup Look-Up-Table to be Word2Vec
        self.pca_mat = None
        print "Initialize LUTs as word2vec and use linear projection layer"

        LUT = np.zeros((self.vs, self.d_w2v), dtype='float32')
        found_words = 0
        for w, v in self.v2i.iteritems():
            if w in self.w2v.vocab:  # all valid words are already in vocab or 'UNK'
                LUT[v] = self.w2v.get_vector(w)
                found_words += 1
            else:
                # LUT[v] = np.zeros((self.d_w2v))
                LUT[v] = self.rng.randn(self.d_w2v)
                LUT[v] = LUT[v] / (np.linalg.norm(LUT[v]) + 1e-6)

        print "Found %d / %d words" %(found_words, len(self.v2i))

        # word 0 is blanked out, word 1 is 'UNK'
        LUT[0] = np.zeros((self.d_w2v))

        # if linear projection layer is not the same shape as LUT, then initialize with PCA
        if self.d_lproj != LUT.shape[1]:
            pca = PCA(n_components=self.d_lproj, whiten=True)
            self.pca_mat = pca.fit_transform(LUT.T)  # 300 x 100?

        # setup LUT!
        self.T_w2v = theano.shared(LUT.astype(theano.config.floatX))

    def build_model(self):
        """Build the memory network pipeline.
        Steps:
            - define memory layer instance
            - define input data
            - encode question
            - add memory layers
            - decode output
            - compute metrics
            - collect paramters, results
        """

        # ------ Define Memory Layer instance
        mem_layer_opts = {'T_w2v': self.T_w2v}
        mem_layer = MemoryLayer(mem_layer_opts)

        # ------ Create Theano parameter holders
        self.t_params = OrderedDict()
        self.t_inputs = OrderedDict()
        self.t_outputs = OrderedDict()

        # ------ Input Data
        T_story = T.itensor3('story')   # batch-size X sentences X words
        T_q = T.imatrix('q')            # batch-size X words
        T_y = T.ivector('y_gt')         # batch-size ('single': word index,  'multi_choice': correct option)
        T_z = T.itensor3('z')           # batch-size X multiple options X words
        T_lr = T.scalar('lr')           # learning rate
        self.t_inputs.update({'T_story': T_story})
        self.t_inputs.update({'T_q': T_q})
        self.t_inputs.update({'T_y': T_y})
        self.t_inputs.update({'T_z': T_z})
        self.t_inputs.update({'T_lr': T_lr})

        # ------ Encode question
        T_w2v_out = self.T_w2v[T_q] * self.T_mask[T_q]
        T_u = T.sum(T_w2v_out, axis=1)
        T_u_norm = T.sqrt(T.sum(T_u ** 2, axis=1))
        T_u = T_u / (T_u_norm.dimshuffle(0, 'x') + 1e-6)

        T_B = theano.shared(init_linear_projection(self.rng, self.d_w2v, self.d_lproj, self.pca_mat), name='B')
        self.t_params.update({'T_B':T_B})
        T_u = T.dot(T_u, T_B)

        # ------ Layers of memory and attention interaction
        for n in range(self.nl):
            # Add one layer of memory
            T_o, T_p = mem_layer.make_layer(self.t_params, T_u, T_story, self.T_mask, self.rng)
            T_u = T_u + T_o

        self.out_debug = T_p

        # ------ Encode multiple choice answers
        T_w2v_out = self.T_w2v[T_z] * self.T_mask[T_z]
        T_g = T.sum(T_w2v_out, axis=2)
        T_g_norm = T.sqrt(T.sum(T_g ** 2, axis=2))
        T_g = T_g / (T_g_norm.dimshuffle(0, 1, 'x') + 1e-6)
        T_g = T.dot(T_g, T_B)

        # ------ Normalize representations before doing dot product
        T_g_norm = T.sqrt(T.sum(T_g ** 2, axis=2))
        T_g2 = T_g / (T_g_norm.dimshuffle(0, 1, 'x') + 1e-6)

        T_u_norm = T.sqrt(T.sum(T_u ** 2, axis=1))
        T_u2 = T_u / (T_u_norm.dimshuffle(0, 'x') + 1e-6)

        # ------ Prediction
        # compute matching score between normalized question (o+u) and answer (g)
        T_h = T_u2.dimshuffle(0, 'x', 1) * T_g2
        T_s = T.sum(T_h, axis=2)

        # ------ Cost: SoftMax, CrossEntropy
        T_yhat = T.nnet.softmax(T_s)
        T_y_pred = T.argmax(T_yhat, axis=1)

        T_lp = T.log(T_yhat)
        T_ll = T_lp[T.arange(T_y.shape[0]), T_y]
        T_cost = T.mean(- T_ll)

        # ------ Collect all results
        self.out_pnorm = T.stack([T.sqrt(T.sum(p ** 2)) for p in self.t_params.values()])  # parameter norms
        print "Model inputs:", self.t_inputs.keys()
        print "Trainable parameters:", self.t_params.keys()
        self.t_outputs.update({'T_cost': T_cost, 'T_yhat': T_yhat, 'T_y_pred': T_y_pred})

    def gradients_and_updates(self, grad_normalize):
        """Compute gradients (t_gparams) using cost and trainable weights (t_params).
        """

        # ------ Compute gradient parameters
        self.t_gparams = OrderedDict({'g_' + k: theano.grad(cost=self.t_outputs['T_cost'], wrt=p)
                                      for k, p in self.t_params.iteritems()})

        # ------ Compute norm and stack it like a vector (to analyze outside)
        # self.out_debug = self.t_gparams['g_T_B']
        self.out_gnorm = T.stack([T.sqrt(T.sum(gp ** 2)) for gp in self.t_gparams.values()])

        # ------ Normalize gradients
        self.g_norm = {}
        if grad_normalize.has_key('max_norm'):      # maximum gradient norm limited
            mn = grad_normalize['max_norm']
            for k in self.t_gparams.keys():
                self.g_norm[k] = T.sqrt(T.sum(self.t_gparams[k] ** 2))
                self.t_gparams[k] = ifel(T.gt(self.g_norm[k], mn),
                                         mn * self.t_gparams[k] / (self.g_norm[k] + 1e-6),
                                         self.t_gparams[k])

        # ------ Update parameters (SGD!)
        self.update_params = []
        for k in self.t_params.keys():
            self.update_params.append([self.t_params[k],
                                       self.t_params[k] - self.t_inputs['T_lr'] * self.t_gparams['g_' + k]])

    def train_function(self):
        """Theano function to train the model using a batch of data.
        """

        train_model = theano.function(inputs=[p for k, p in self.t_inputs.iteritems()],
                                      outputs=[self.t_outputs['T_cost'], self.t_outputs['T_yhat'],
                                               self.out_gnorm, self.out_pnorm],
                                      # mode=theano.compile.DebugMode,
                                      # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
                                      # mode='FAST_COMPILE',
                                      updates=self.update_params,
                                      on_unused_input='warn')
        return train_model

    def test_function(self):
        """Theano function to test the model on a batch of data.
        """
        ignore_inputs = ['T_lr', 'T_y']
        test_model = theano.function(inputs=[p for k, p in self.t_inputs.iteritems() if k not in ignore_inputs],
                                     outputs=self.t_outputs['T_yhat'],
                                     on_unused_input='warn')
        return test_model

    def save_model(self, save_filename):
        """Save model parameters.
        """
        np.save(save_filename, self.t_params['T_B'].eval())


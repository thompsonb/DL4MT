# -*- coding: utf-8 -*-

"""
Test nematusLL for consistency with nematus
"""

import os
import unittest
import sys
import numpy as np
import logging
import Pyro4

nem_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
sys.path.insert(1, nem_path)
from nematus.pyro_utils import setup_remotes, get_random_key, get_unused_port
from nematus.util import load_dict

from unit_test_utils import initialize

GPU_ID = 0
VOCAB_SIZE = 90000
SRC = 'ro'
TGT = 'en'
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')

model_options = dict(factors=1,  # input factors
                     dim_per_factor=None,
                     # list of word vector dimensionalities (one per factor): [250,200,50] for dimensionality of 500
                     encoder='gru',
                     decoder='gru_cond',
                     patience=10,  # early stopping patience
                     max_epochs=5000,
                     finish_after=10000000,  # finish after this many updates
                     map_decay_c=0.,  # L2 regularization penalty towards original weights
                     alpha_c=0.,  # alignment regularization
                     shuffle_each_epoch=True,
                     finetune=False,
                     finetune_only_last=False,
                     sort_by_length=True,
                     use_domain_interpolation=False,
                     domain_interpolation_min=0.1,
                     domain_interpolation_inc=0.1,
                     domain_interpolation_indomain_datasets=['indomain.en', 'indomain.fr'],
                     maxibatch_size=20,  # How many minibatches to load at one time
                     model_version=0.1,  # store version used for training for compatibility
                     pyro_key=None,  # pyro hmac key
                     pyro_port=None,  # pyro nameserver port
                     pyro_name=None,  # if None, will import instead of assuming a server is running
                     saveto='model.npz',
                     reload_=True,
                     dim_word=500,
                     dim=1024,
                     n_words=VOCAB_SIZE,
                     n_words_src=VOCAB_SIZE,
                     decay_c=0.,
                     clip_c=1.,
                     lrate=0.0001,
                     optimizer='adadelta',
                     maxlen=50,
                     batch_size=80,
                     valid_batch_size=80,
                     datasets=[DATA_DIR + '/corpus.bpe.' + SRC, DATA_DIR + '/corpus.bpe.' + TGT],
                     valid_datasets=[DATA_DIR + '/newsdev2016.bpe.' + SRC, DATA_DIR + '/newsdev2016.bpe.' + TGT],
                     dictionaries=[DATA_DIR + '/corpus.bpe.' + SRC + '.json',
                                   DATA_DIR + '/corpus.bpe.' + TGT + '.json'],
                     validFreq=10000,
                     dispFreq=10,
                     saveFreq=30000,
                     sampleFreq=10000,
                     use_dropout=False,
                     dropout_embedding=0.2,  # dropout for input embeddings (0: no dropout)
                     dropout_hidden=0.2,  # dropout for hidden layers (0: no dropout)
                     dropout_source=0.1,  # dropout source words (0: no dropout)
                     dropout_target=0.1,  # dropout target words (0: no dropout)
                     overwrite=False,
                     external_validation_script='./validate.sh',
                     )

x0 = np.array([[[3602],
                [8307],
                [7244],
                [7],
                [58],
                [9],
                [5893],
                [62048],
                [11372],
                [4029],
                [25],
                [34],
                [2278],
                [5],
                [4266],
                [11],
                [2852],
                [3],
                [2298],
                [2],
                [23912],
                [6],
                [16358],
                [3],
                [730],
                [2328],
                [5],
                [28],
                [353],
                [4],
                [0], ]])  # 0 = EOS

xx0 = np.tile(x0, [1, 1, 2])

x1 = np.array([[[3602],
                [8307],
                [7244],
                [7],
                [58],
                [9],
                [4],
                [0], ]])  # 0 = EOS

# False Truth for testing Gradients
y1 = np.array([[[1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [0], ]])  # 0 = EOS

x_0 = np.array([[[2], [0], [0], [0]]])

x_0_mask = np.array([[1], [1], [1], [0]], dtype=np.float32)

y_0 = np.array([[17], [9], [41], [120], [7], [117], [1087], [476], [1715], [62], [2], [0]])

y_0_mask = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype=np.float32)

per_sample_weight_1 = np.array([1, 1], dtype=np.float32)

xx_0 = np.array([[[4, 2], [4, 0], [3, 0], [2, 0]]])

xx_0_mask = np.array([[2, 1], [0, 1], [0, 1], [0, 0]], dtype=np.float32)

yy_0 = np.array([[22, 17], [24, 9], [31, 41], [420, 120], [37, 7], [127, 117], [1387, 1087], [446, 476], [1515, 1715],
                 [22, 62], [12, 2], [0, 0]])

yy_0_mask = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
                     dtype=np.float32)

per_sample_weight_2 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

xx_1 = np.tile(xx_0, [2])

yy_1 = np.tile(yy_0, [2])

xx_1_mask = np.tile(xx_0_mask, [2])

yy_1_mask = np.tile(yy_0_mask, [2])

per_sample_weight = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

x_1 = np.array([[[2, 2, 2, 2],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]])
x_1_mask = np.array([[1., 1., 1., 1.],
                     [1., 1., 1., 1.],
                     [1., 1., 1., 1.],
                     [0., 0., 0., 0.]], dtype=np.float32)
y_1 = np.array([[17, 17, 17, 17],
                [9, 9, 9, 9],
                [41, 41, 41, 41],
                [120, 120, 120, 120],
                [7, 7, 7, 7],
                [117, 117, 117, 117],
                [1087, 1087, 1087, 1087],
                [476, 476, 476, 476],
                [1715, 1715, 1715, 1715],
                [62, 62, 62, 62],
                [2, 2, 2, 2],
                [0, 0, 0, 0]])
y_1_mask = np.array([[1., 1., 1., 1.],
                     [1., 1., 1., 1.],
                     [1., 1., 1., 1.],
                     [1., 1., 1., 1.],
                     [1., 1., 1., 1.],
                     [1., 1., 1., 1.],
                     [1., 1., 1., 1.],
                     [1., 1., 1., 1.],
                     [1., 1., 1., 1.],
                     [1., 1., 1., 1.],
                     [1., 1., 1., 1.],
                     [1., 1., 1., 1.]], dtype=np.float32)


class PyroNematusTests(unittest.TestCase):
    # Forces the class to have these fields.
    logger = None
    context_mgr = None

    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger(__file__)
        cls.logger.info("========================================================================================")
        cls.logger.info("Setting up the pyro remote server as well as the nematus instance to test consistency.")
        cls.logger.info("========================================================================================")
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        remote_script = os.path.join(current_script_dir, '../nematus/nmt_remote.py')
        pyro_name = 'remote'

        pyro_key = get_random_key()
        pyro_port = get_unused_port()

        cls.context_mgr = setup_remotes(remote_metadata_list=[dict(script=remote_script,
                                                                   name=pyro_name, gpu_id=GPU_ID)],
                                        pyro_port=pyro_port,
                                        pyro_key=pyro_key)
        cls.context_mgr.__enter__()
        cls.remote_interface = initialize(model_options=model_options,
                                          pyro_port=pyro_port,
                                          pyro_name=pyro_name,
                                          pyro_key=pyro_key)

    @classmethod
    def tearDownClass(cls):
        cls.logger.info("========================================================================================")
        cls.logger.info("Tearing down the pyro remote server as well as the nematus instance")
        cls.logger.info("========================================================================================")
        cls.context_mgr.__exit__(None, None, None)

    def assert_params_same(self, params1, params2):
        k1 = params1.keys()
        k2 = params2.keys()
        k1.sort()
        k2.sort()
        self.assertTrue(k1 == k2)
        for k in k1:
            self.assertTrue(np.allclose(params1[k], params2[k]), 'value for %s should match' % k)

    def assert_params_different(self, params1, params2):
        k1 = params1.keys()
        k2 = params2.keys()
        k1.sort()
        k2.sort()
        self.assertTrue(k1 == k2)
        diff_vec = [not np.allclose(params1[k], params2[k]) for k in k1]
        self.logger.info("Difference vector is: " + str(diff_vec))
        self.assertTrue(any(diff_vec))

    def test_f_init_dims(self):
        """
        Best I can tell, f_init is only ever given one sentence, but it appears to be
        written to process multiple sentences.
        """
        self.logger.info("========================================================================================")
        self.logger.info("Starting the f_init_dims test to determine that x_f_init acts as expected.")
        self.logger.info("========================================================================================")
        x0_state0, x0_ctx0 = self.remote_interface.x_f_init(x0)  # (1, 1024) (31, 1, 2048)

        # If tile input, state/context should be tiled too
        xx0_state0, xx0_ctx0 = self.remote_interface.x_f_init(xx0)  # (2, 1024) (31, 2, 2048)
        self.assertTrue(np.allclose(np.tile(x0_state0, [2, 1]), xx0_state0))
        self.assertTrue(np.allclose(np.tile(x0_ctx0, [1, 2, 1]), xx0_ctx0))

        # Different inputs should create different state
        x1_state0, x1_ctx0 = self.remote_interface.x_f_init(x1)
        self.assertFalse(np.allclose(x0_state0, x1_state0))

        # Different inputs (of same length) should create different state and context
        x1_2_state0, x1_2_ctx0 = self.remote_interface.x_f_init(x1 * 2)
        self.assertFalse(np.allclose(x1_state0, x1_2_state0))
        self.assertFalse(np.allclose(x1_ctx0, x1_2_ctx0))

    def test_f_next_dims(self):
        self.logger.info("========================================================================================")
        self.logger.info("Starting the f_next_dims test to determine that x_f_next acts as expected.")
        self.logger.info("========================================================================================")
        self.remote_interface.set_noise_val(0)
        x0_state0, x0_ctx0 = self.remote_interface.x_f_init(x0)
        x0_prob1, x0_word1, x0_state1 = self.remote_interface.x_f_next(np.array([2893, ]), x0_ctx0, x0_state0)
        x0_prob2, x0_word2, x0_state2 = self.remote_interface.x_f_next(np.array([9023, ]), x0_ctx0, x0_state1)
        self.assertFalse(np.allclose(x0_state0, x0_state1), 'state should be changing')
        self.assertFalse(np.allclose(x0_prob1, x0_prob2), 'probability should be changing')
        # word might not change...

        self.logger.info('x0 prob shape, ' + str(x0_prob1.shape))
        self.logger.info('x0 word shape, ' + str(x0_word1.shape))
        self.logger.info('x0 state shape, ' + str(x0_state2.shape))

        xx0_state0, xx0_ctx0 = self.remote_interface.x_f_init(xx0)
        xx0_prob1, xx0_word1, xx0_state1 = self.remote_interface.x_f_next(np.array([2893, 2893]), xx0_ctx0, xx0_state0)
        xx0_prob2, xx0_word2, xx0_state2 = self.remote_interface.x_f_next(np.array([9023, 9023]), xx0_ctx0, xx0_state1)

        self.logger.info('xx0 prob shape, ' + str(xx0_prob1.shape))
        self.logger.info('xx0 word shape, ' + str(xx0_word1.shape))
        self.logger.info('xx0 state shape, ' + str(xx0_state2.shape))

        self.assertTrue(np.allclose(np.tile(x0_prob1, [2, 1]), xx0_prob1))
        self.assertTrue(np.allclose(np.tile(x0_prob2, [2, 1]), xx0_prob2))
        # jitter??
        # print 'same??', np.tile(x0_word1, [2]), xx0_word1
        # self.assertTrue(np.allclose(np.tile(x0_word1, [2]), xx0_word1))
        # self.assertTrue(np.allclose(np.tile(x0_word2, [2]), xx0_word2))
        self.assertTrue(np.allclose(np.tile(x0_state1, [2, 1]), xx0_state1))
        self.assertTrue(np.allclose(np.tile(x0_state2, [2, 1]), xx0_state2))

    def test_f_init_side_effects(self):
        self.logger.info("========================================================================================")
        self.logger.info("Starting a side effect test to determine that f_init has no side effects. ")
        self.logger.info("========================================================================================")
        self.remote_interface.x_f_update(lrate=0.1)  # should zero grads
        params0 = self.remote_interface.get_params_from_theano()
        _, _ = self.remote_interface.x_f_init(x0)
        _, _ = self.remote_interface.x_f_init(x1)
        _, _ = self.remote_interface.x_f_init(x0)
        params2 = self.remote_interface.get_params_from_theano()
        self.remote_interface.x_f_update(lrate=0.1)  # grads should still be zero, so this should not change params
        params1 = self.remote_interface.get_params_from_theano()
        self.assert_params_same(params0, params2)
        self.assert_params_different(params0, params1)

    def test_get_params_consistency(self):
        self.logger.info("========================================================================================")
        self.logger.info("Starting the param_consistency test to determine that params do not change unexpectedly.")
        self.logger.info("========================================================================================")
        params0 = self.remote_interface.get_params_from_theano()
        params1 = self.remote_interface.get_params_from_theano()
        params2 = self.remote_interface.get_params_from_theano()
        self.assert_params_same(params0, params1)
        self.assert_params_same(params0, params2)

        self.remote_interface.x_f_update(lrate=0.1)  # should zero grads
        params3 = self.remote_interface.get_params_from_theano()
        self.remote_interface.x_f_update(lrate=0.1)  # grads=0, so this should not change params
        params4 = self.remote_interface.get_params_from_theano()
        self.assert_params_different(params3, params4)

    def test_f_grad_shared(self):
        self.logger.info("========================================================================================")
        self.logger.info("Starting the f_next_dims test to determine that x_f_next acts as expected.")
        self.logger.info("========================================================================================")
        self.remote_interface.set_noise_val(0)
        params = self.remote_interface.get_params_from_theano()
        grad = self.remote_interface.x_f_grad_shared(x_0, x_0_mask, y_0, y_0_mask, per_sent_weight=[1])
        self.remote_interface.send_params_to_theano(params)
        grad2 = self.remote_interface.x_f_grad_shared(x_1, x_1_mask, y_1, y_1_mask, per_sent_weight=per_sample_weight)
        self.assertAlmostEqual(grad, grad2, places=1)
        self.remote_interface.send_params_to_theano(params)
        grad3 = self.remote_interface.x_f_grad_shared(xx_0, xx_0_mask, yy_0, yy_0_mask,
                                                      per_sent_weight=per_sample_weight_1)
        self.remote_interface.send_params_to_theano(params)
        grad4 = self.remote_interface.x_f_grad_shared(xx_1, xx_1_mask, yy_1, yy_1_mask,
                                                      per_sent_weight=per_sample_weight_2)
        self.assertAlmostEqual(grad3, grad4, places=1)


if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-

import os
import unittest
import sys
import numpy as np
import logging
import Pyro4
import codecs

from copy import deepcopy
import json
import numpy

nem_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
sys.path.insert(1, nem_path)
from nematus.pyro_utils import setup_remotes, get_random_key, get_unused_port
from nematus.util import load_dict
from nematus.nmt_client import default_model_options
from nematus.nmt_utils import gen_sample, gen_par_sample, prepare_data
from nematus.config import wmt16_systems_dir

from unit_test_utils import initialize

# TODO: copy code!
def build_model_options(model_dir, lang0, lang1):
    model_options = deepcopy(default_model_options)
    model_options.update(json.loads(open(os.path.join(model_dir, 'model.npz.json')).read()))
    model_options['saveto'] = os.path.join(model_dir, 'model.npz')
    model_options['dictionaries'] = [os.path.join(model_dir, 'vocab.%s.json' % lang0),
                                     os.path.join(model_dir, 'vocab.%s.json' % lang1)]
    # TODO: other paths... datasets, valid_datasets?
    return model_options


LANG_A = 'en'
LANG_B = 'de'
modeldir_AB = os.path.join(wmt16_systems_dir, '%s-%s'%(LANG_A, LANG_B))

dictionaries = [modeldir_AB + '/vocab.%s.json' % LANG_A, modeldir_AB + '/vocab.%s.json' % LANG_B],
model_options = build_model_options(modeldir_AB, LANG_A, LANG_B)

GPU_ID = 0


def sample_par(lines, model_options, f_init, f_next, beam_size=3, suppress_unk=True):
        dictionaries = model_options['dictionaries']
        dictionaries_source = dictionaries[:-1]
        dictionary_target = dictionaries[-1]

        # load source dictionary and invert
        word_dicts = []
        word_idicts = []
        for dictionary in dictionaries_source:
            word_dict = load_dict(dictionary)
            if model_options['n_words_src']:
                for key, idx in word_dict.items():
                    if idx >= model_options['n_words_src']:
                        del word_dict[key]
            word_idict = dict()
            for kk, vv in word_dict.iteritems():
                word_idict[vv] = kk
            word_idict[0] = '<eos>'
            word_idict[1] = 'UNK'
            word_dicts.append(word_dict)
            word_idicts.append(word_idict)

        # load target dictionary and invert
        word_dict_trg = load_dict(dictionary_target)
        word_idict_trg = dict()
        for kk, vv in word_dict_trg.iteritems():
            word_idict_trg[vv] = kk
        word_idict_trg[0] = '<eos>'
        word_idict_trg[1] = 'UNK'

        def _seqs2words(cc):
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict_trg[w])
            return ' '.join(ww).replace('@@ ', '')


        seqs = []
        for idx, line in enumerate(lines):
            words = line.strip().split()

            x = []
            for w in words:
                w = [word_dicts[i][f] if f in word_dicts[i] else 1 for (i, f) in enumerate(w.split('|'))]
                if len(w) != model_options['factors']:
                    raise Exception('Error: expected {0} factors, but input word has {1}\n'.format(model_options['factors'], len(w)))
                x.append(w)

            x += [[0] * model_options['factors']]
            seqs.append(x)

        seqs_y_dummy = [[x[0] for x in element] for element in seqs]
        sequences, xmask, dummy_y, ymask = prepare_data(seqs, seqs_y_dummy)

        print 'Calling translate_par'
        parsample, parscore, parword_probs = gen_par_sample([f_init, ], [f_next, ],
                                                   sequences, xmask,                                             
                                                   k=beam_size, maxlen=200,
                                                   suppress_unk=suppress_unk)

        compare_samples = []
        for i in range(len(seqs)):
            mask_size = int(round(np.sum(xmask[:,i])))
            seq = sequences[:, :mask_size, i:i+1]
            sample, score, word_probs, _, _ = gen_sample([f_init, ], [f_next, ], seq, k=beam_size, maxlen=200, stochastic = False, suppress_unk=suppress_unk)
            compare_samples += sample

        sample_words = []
        for sents in parsample:
            #sample_words.append([_seqs2words(cand) for cand in sents])
            sample_words += sents

        return sample_words, compare_samples


class ParallelSampleTestCase(unittest.TestCase):
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
        cls.logger.info("todo")
        cls.logger.info("========================================================================================")
        cls.context_mgr.__exit__(None, None, None)


    def test_parallel_sample(self):
        #lines = ['this is a sentence', 'this is some other sentence']
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        with codecs.open(os.path.join(current_script_dir, 'test_data/par_samp_test')) as fh:
            lines = [line for line in fh]
        parsamples, nonparsamples = sample_par(lines=lines,
                                                             model_options=model_options, 
                                                             f_init = self.remote_interface.x_f_init,
                                                             f_next = self.remote_interface.x_f_next,
                                                             beam_size=3, 
                                                             suppress_unk=True)
        self.assertEqual(parsamples, nonparsamples)
        #for line, sents, scores_per_sent in zip(lines, sample_words, score):
        #    print '------------------', line
        #    for sentX, scoreX in zip(sents, scores_per_sent):
        #        print 'Score %f: %s'%(scoreX, sentX)

        
if __name__ == '__main__':
    unittest.main()

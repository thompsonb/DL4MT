#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a neural machine translation model with soft attention
"""

import copy
import os
import sys
import time
from copy import deepcopy

import Pyro4
import numpy
from theano.tensor.shared_randomstreams import RandomStreams

from data_iterator import TextIterator, MonoIterator
from nmt_client import default_model_options
from nmt_utils import prepare_data, gen_sample
from pyro_utils import setup_remotes, get_random_key, get_unused_port
from util import load_dict

profile = False
bypass_pyro = False  # True
LOCALMODELDIR = '' # TODO: add language model directory


def _add_dim(x_pre):
    # add an extra dimension, as expected on x input to prepare_data
    # TODO: assuming no factors!
    xx = []
    for sent in x_pre:
        xx.append([[wid, ] for wid in sent])
    return xx


def _train_foo(remote_mt, _xxx, _yyy, _per_sent_weight, _lrate, maxlen):
    _x_prep, _x_mask, _y_prep, _y_mask = prepare_data(_add_dim(_xxx), _yyy, maxlen=maxlen)
    if _x_prep is None:
        return None
    remote_mt.set_noise_val(0.)
    # returns cost, which is related to log probs BUT may be weighted per sentence, and may include regularization terms!
    cost = remote_mt.x_f_grad_shared(_x_prep, _x_mask, _y_prep, _y_mask, _per_sent_weight, per_sent_cost=True)
    remote_mt.x_f_update(_lrate)  # TODO: WAIT TILL END?
    # check for bad numbers, usually we remove non-finite elements
    # and continue training - but not done here
    if any(numpy.isnan(cost)) or any(numpy.isinf(cost)):
        raise Exception('NaN detected')
    # TODO: this is wasteful! save time and compute at same time as cost above
    per_sent_neg_log_prob = remote_mt.x_f_log_probs(_x_prep, _x_mask, _y_prep, _y_mask, )
    # log(prob) is negative; higher is better, i.e. this is a reward
    # -log(prob) is positivel smaller is better, i.e. this is a cost
    # scale by -1. to get back to a reward
    per_sent_mt_reward = -1.0 * per_sent_neg_log_prob
    return per_sent_mt_reward


def monolingual_train(mt_systems, lm_1, 
                      data, trng, k, maxlen, 
                      worddicts_r, worddicts, 
                      alpha, learning_rate_big,
                      learning_rate_small):

    mt_01, mt_10 = mt_systems
    num2word_01, num2word_10 = worddicts_r
    word2num_01, word2num_10 = worddicts

    for sent in data:
        print '#'*20, 'NEW SENTENCE'

        try:
            print 'sent 0:', ' '.join([num2word_01[0][foo[0]] for foo in sent])
        except:
            print 'could not print sent 0'

        # TRANSLATE 0->1
        sents1_01, scores_1, _, _, _ = gen_sample([mt_01.x_f_init],
                                                  [mt_01.x_f_next],
                                                  numpy.array([sent, ]),
                                                  trng=trng, k=k,
                                                  maxlen=maxlen,
                                                  stochastic=False,
                                                  argmax=False,
                                                  suppress_unk=True,
                                                  return_hyp_graph=False)

        try:
            for ii, sent1_01 in enumerate(sents1_01):
                print 'sent 0->1 #%d (in system 01 vocab):'%ii, ' '.join([num2word_01[1][foo] for foo in sent1_01])
        except:
            print 'failed to print sent 0->1 sentences (in system 01 vocab)'

        # strip out <eos>, </s> tags (I have no idea where </s> is coming from!)
        sents1_01_tmp = []
        for sent_1 in sents1_01:
            sents1_01_tmp.append([x for x in sent_1 if num2word_01[1][x] not in ('<eos>', '</s>')])
        sents1_01 = sents1_01_tmp

        # Clean Data (for length - translated sentence may not be acceptable length)
        sents1_01_clean = []
        sents0_01_clean = []
        for ii, sent_1 in enumerate(sents1_01):
            if len(sent_1) < 2:
                print 'len(sent #%d)=%d, < 2. skipping'%(ii, len(sent_1))
            elif len(sent_1) >= maxlen:
                print 'len(sent #%d)=%d, > %d. skipping'%(ii, len(sent_1), maxlen)
            else:
                sents1_01_clean.append(sent_1)
                sents0_01_clean.append([x[0] for x in sent])

        if len(sents1_01_clean) == 0:
            print "No acceptable length data out of 0->1 system"
            continue

        # LANGUAGE MODEL SCORE IN LANG 1        
        r_1 = lm_1.score(numpy.array(sents1_01_clean).T)
        print "scores_lm1", r_1

        # The two MT systems have different vocabularies
        # Convert from mt01's vocab to mt10's vocab
        sents1_10_clean = []
        for num1 in sents1_01_clean:
            words = [num2word_01[1][num] for num in num1]
            words = [w for w in words if w not in ('<eos>', '</s>')]
            num2 = [word2num_10[0][word] for word in words]
            sents1_10_clean.append(num2)


        try:
            for ii, sent1_01 in enumerate(sents1_10_clean):
                print 'sent 0->1 (in system 10 vocab) #%d:'%ii, ' '.join([num2word_10[0][foo] for foo in sent1_01])
        except:
            print 'failed to print sent 0->1 sentences (in system 10 vocab)'


        sents0_10_clean = []
        for num1 in sents0_01_clean:
            words = [num2word_01[0][num] for num in num1]
            words = [w for w in words if w not in ('<eos>', '</s>')]
            num2 = [word2num_10[1][word] for word in words]
            sents0_10_clean.append(num2)


        try:
            for ii, sent0_10 in enumerate(sents0_10_clean):
                print 'sent 0 (in system 10 vocab) #%d:'%ii, ' '.join([num2word_10[1][foo] for foo in sent0_10])
        except:
            print 'failed to print sent 0 sentences (in system 10 vocab)'


        # MT 0->1 SCORE AND UPDATE
        sents0_10, _, _, _, _ = gen_sample([mt_10.x_f_init],
                                           [mt_10.x_f_next],
                                           numpy.array([[[x, ] for x in sents1_10_clean[0]], ]),
                                           trng=trng, k=1,
                                           maxlen=maxlen,
                                           stochastic=False,
                                           argmax=False,
                                           suppress_unk=True,
                                           return_hyp_graph=False)

        try:
            print '[just for debug] first sentence from 0->1->0 (in system 10 vocab)', ' '.join([num2word_10[1][x] for x in sents0_10[0]])
        except:
            print 'failed to print 0->1->0 sentences'

        per_sent_weight = [(1 - alpha) / k for _ in sents1_01_clean]

        print 'psw10=', per_sent_weight

        r_2 = _train_foo(mt_10, sents1_10_clean, sents0_10_clean,
                         per_sent_weight, learning_rate_big, maxlen)

        if r_2 is None:  # failed due to assert
            print 'WARNING: data prep failed (_x_prep is None). ignoring...'
            continue
            

        print 'reward_mt10', r_2

        per_sent_weight = [-1 * (alpha * s1 + (1 - alpha) * s2) / k for s1, s2 in zip(r_1, r_2)]
        print 'psw01=', per_sent_weight

        final_r = _train_foo(mt_01, sents0_10_clean, sents1_10_clean, per_sent_weight,
                             learning_rate_small, maxlen)

        print final_r

    return 1


def few_dict_items(a):
    return [(x, a[x]) for x in list(a)[:15]], 'len=%d'%len(a)


def check_model_options(model_options, dictionaries):
    if model_options['dim_per_factor'] is None:
        if model_options['factors'] == 1:
            model_options['dim_per_factor'] = [model_options['dim_word']]
        else:
            sys.stderr.write('Error: if using factored input, you must specify \'dim_per_factor\'\n')
            sys.exit(1)

    assert (len(dictionaries) == model_options['factors'] + 1)  # one dictionary per source factor + 1 for target factor
    assert (len(model_options['dim_per_factor']) == model_options[
        'factors'])  # each factor embedding has its own dimensionality
    assert (sum(model_options['dim_per_factor']) == model_options[
        'dim_word'])  # dimensionality of factor embeddings sums up to total dimensionality of input embedding vector

    if model_options['factors'] > 1:
        raise Exception('I probably broke factors...')


def reverse_model_options(model_options_a_b):
    model_options_b_a = copy.deepcopy(model_options_a_b)
    # for key in ['datasets', 'valid_datasets', 'domain_interpolation_indomain_datasets']:
    #    model_options_b_a[key] = model_options_a_b[key][::-1]
    model_options_b_a['saveto'] = model_options_a_b['saveto'] + '.BA'
    model_options_b_a['n_words_src'] = model_options_a_b['n_words']  # n_words is target
    model_options_b_a['n_words'] = model_options_a_b['n_words_src']
    return model_options_b_a


def train(**kwargs):
    if bypass_pyro:
        train2(**kwargs)
    else:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        nmt_remote_script = os.path.join(current_script_dir, 'nmt_remote.py')
        lm_remote_script = os.path.join(current_script_dir, 'lm_remote.py')

        kwargs.update({'pyro_name_mt_a_b': 'mtAB',
                       'pyro_name_mt_b_a': 'mtBA',
                       'pyro_name_lm_a': 'lmA',
                       'pyro_name_lm_b': 'lmB',
                       'pyro_key': get_random_key(),
                       'pyro_port': get_unused_port(),
                       })

        with setup_remotes(
                remote_metadata_list=[dict(script=nmt_remote_script, name=kwargs['pyro_name_mt_a_b'], gpu_id=kwargs['mt_gpu_ids'][0]),
                                      dict(script=nmt_remote_script, name=kwargs['pyro_name_mt_b_a'], gpu_id=kwargs['mt_gpu_ids'][1]),
                                      dict(script=lm_remote_script, name=kwargs['pyro_name_lm_a'], gpu_id=kwargs['lm_gpu_ids'][0]),
                                      dict(script=lm_remote_script, name=kwargs['pyro_name_lm_b'], gpu_id=kwargs['lm_gpu_ids'][1])],
                pyro_port=kwargs['pyro_port'],
                pyro_key=kwargs['pyro_key']):
            train2(**kwargs)


# noinspection PyUnusedLocal
def train2(model_options_a_b=None,
           model_options_b_a=None,
           max_epochs=5000,
           finish_after=10000000,
           disp_freq=100,
           save_freq=1000,
           lrate=0.01,
           maxlen=100,
           batch_size=16,
           valid_batch_size=16,
           patience=10,
           parallel_datasets=(),
           valid_datasets=(),
           monolingual_datasets=(),
           dictionaries_a_b=None,
           dictionaries_b_a=None,
           #domain_interpolation_indomain_datasets=('indomain.en', 'indomain.fr'), # TODO unused
           valid_freq=1000,
           sample_freq=100,
           overwrite=False,
           external_validation_script=None,
           shuffle_each_epoch=True,
           sort_by_length=True,
           use_domain_interpolation=False,
           domain_interpolation_min=0.1,
           domain_interpolation_inc=0.1,
           maxibatch_size=20,
           pyro_key=None,
           pyro_port=None,
           pyro_name_mt_a_b=None,
           pyro_name_mt_b_a=None,
           pyro_name_lm_a=None,
           pyro_name_lm_b=None,
           language_models=(),
           mt_gpu_ids=(),
           lm_gpu_ids=(),
           ):

    if model_options_a_b is None:
        model_options_a_b = default_model_options

    if model_options_b_a is None:
        model_options_b_a = reverse_model_options(model_options_a_b)

    check_model_options(model_options_a_b, dictionaries_a_b)
    check_model_options(model_options_b_a, dictionaries_b_a)

    # I can't come up with a reason that trng must be shared... (I use a different seed)
    trng = RandomStreams(hash(__file__) % 4294967294)

    def create_worddicts_and_update_model_options(dictionaries, model_opts):
        # load dictionaries and invert them
        worddicts = [None] * len(dictionaries)
        worddicts_r = [None] * len(dictionaries)
        for k, dd in enumerate(dictionaries):
            worddicts[k] = load_dict(dd)
            worddicts_r[k] = dict()
            for kk, vv in worddicts[k].iteritems():
                worddicts_r[k][vv] = kk
            worddicts_r[k][0] = '<eos>'
            worddicts_r[k][1] = 'UNK'

        if model_opts['n_words_src'] is None:
            model_opts['n_words_src'] = len(worddicts[0])
        if model_opts['n_words'] is None:
            model_opts['n_words'] = len(worddicts[1])

        return worddicts, worddicts_r

    worddicts_a_b, worddicts_r_a_b = create_worddicts_and_update_model_options(dictionaries_a_b, model_options_a_b)
    worddicts_b_a, worddicts_r_b_a = create_worddicts_and_update_model_options(dictionaries_b_a, model_options_b_a)
    

    print '############################'
    print 'len(r_a_b)', len(worddicts_r_a_b),
    print 'r_a_b[0]', few_dict_items(worddicts_r_a_b[0]), '...'
    print 'r_a_b[1]', few_dict_items(worddicts_r_a_b[1]), '...'
    print 'r_b_a[0]', few_dict_items(worddicts_r_b_a[0]), '...'
    print 'r_b_a[1]', few_dict_items(worddicts_r_b_a[1]), '...'
    #print type(worddicts_a_b)
    #print 'len(a_b)', len(worddicts_a_b)


    def _load_data(dataset_a,
                   dataset_b,
                   valid_dataset_a,
                   valid_dataset_b,
                   dict_a,
                   dict_b,
                   model_opts):
        _train = TextIterator(dataset_a, dataset_b,
                              dict_a, dict_b,
                              n_words_source=model_opts['n_words_src'],
                              n_words_target=model_opts['n_words'],
                              batch_size=batch_size,
                              maxlen=maxlen,
                              skip_empty=True,
                              shuffle_each_epoch=shuffle_each_epoch,
                              sort_by_length=sort_by_length,
                              maxibatch_size=maxibatch_size)

        if valid_datasets and valid_freq and False:
            _ = TextIterator(valid_dataset_a, valid_dataset_b,
                             dict_a, dict_b,
                             n_words_source=model_opts['n_words_src'],
                             n_words_target=model_opts['n_words'],
                             batch_size=valid_batch_size,
                             maxlen=maxlen)

        return _train  # , _valid

    def _load_mono_data(dataset,
                        valid_dataset,
                        dict_list,
                        model_opts):
        _train = MonoIterator(dataset,
                              dict_list,
                              n_words_source=model_opts['n_words_src'],
                              batch_size=batch_size,
                              maxlen=maxlen,
                              skip_empty=True,
                              shuffle_each_epoch=shuffle_each_epoch,
                              sort_by_length=sort_by_length,
                              maxibatch_size=maxibatch_size)

        if valid_datasets and valid_freq and False:
            _ = MonoIterator(valid_dataset,
                             dict_list,
                             n_words_source=model_opts['n_words_src'],
                             batch_size=valid_batch_size,
                             maxlen=maxlen)

        return _train  # , _valid

    print 'Loading data'
    domain_interpolation_cur = None

    train_a_b = _load_data(parallel_datasets[0], parallel_datasets[1], {}, {},  # TODO: Placeholders until we have fixed the valid_dataset
                           [dictionaries_a_b[0], ], dictionaries_a_b[1], model_options_a_b)  # TODO: why a list?

    train_b_a = _load_data(parallel_datasets[1], parallel_datasets[0], {}, {},
                           [dictionaries_b_a[0], ], dictionaries_b_a[1], model_options_b_a)  # TODO: why a list?
    train_a = _load_mono_data(monolingual_datasets[0], {}, (dictionaries_a_b[0],), model_options_a_b)
    train_b = _load_mono_data(monolingual_datasets[1], {}, (dictionaries_b_a[0],), model_options_b_a)

    def data_generator(data_a_b, data_b_a, mono_a, mono_b):
        #    def data_generator(data_a_b, data_b_a, mono):
        while True:
            ab_a, ab_b = data_a_b.next()
            ba_b, ba_a = data_b_a.next()
            a = mono_a.next()
            b = mono_b.next()
            yield 'mt', ((ab_a, ab_b), (ba_b, ba_a))
            yield 'mono-a', a  
            yield 'mono-b', b

    training = data_generator(train_a_b, train_b_a, train_a, train_b)
    #    training = data_generator(train_a_b, train_b_a, monoAB)

    # In order to transfer numpy objects across the network, must use pickle as Pyro Serializer.
    # Also requires various environment flags (PYRO_SERIALIZERS_ACCEPTED, PYRO_SERIALIZER)
    #   for both name server and server.
    Pyro4.config.SERIALIZER = 'pickle'
    Pyro4.config.NS_PORT = pyro_port

    if (pyro_name_mt_a_b is not None) and (pyro_name_mt_b_a is not None):
        print 'Setting up remote translation engines'
        remote_mt_a_b = Pyro4.Proxy("PYRONAME:{0}".format(pyro_name_mt_a_b))
        remote_mt_a_b._pyroHmacKey = pyro_key
        remote_mt_b_a = Pyro4.Proxy("PYRONAME:{0}".format(pyro_name_mt_b_a))
        remote_mt_b_a._pyroHmacKey = pyro_key
    else:  # better for IDE
        print 'Importing translation engines'
        from nmt_remote import RemoteMT
        remote_mt_a_b = RemoteMT()
        remote_mt_b_a = RemoteMT()

    if (pyro_name_lm_a is not None) and (pyro_name_lm_b is not None):
        print 'Setting up remote language models'
        remote_lm_a = Pyro4.Proxy("PYRONAME:{0}".format(pyro_name_lm_a))
        remote_lm_a._pyroHmacKey = pyro_key
        remote_lm_b = Pyro4.Proxy("PYRONAME:{0}".format(pyro_name_lm_b))
        remote_lm_b._pyroHmacKey = pyro_key
    else:  # better for IDE
        print 'Importing language models'
        from lm_remote import RemoteLM
        remote_lm_a = RemoteLM()
        remote_lm_b = RemoteLM()

    print 'initializing remote MT0'
    remote_mt_a_b.init(model_options_a_b)
    print 'initializing remote MT1'
    # remote_mt_a_b.set_noise_val(0.5) # TEST: make sure it initilized
    remote_mt_b_a.init(model_options_b_a)
    print 'initializing remote LM0'
    remote_lm_a.init(language_models[0], worddicts_r_b_a[1])  # scoring going INTO language, so use A from BA
    print 'initializing remote LM1'
    remote_lm_b.init(language_models[1], worddicts_r_a_b[1])  # scoring going INTO language, so use B from AB

    # # setup asynchronous remote wrappers
    # remote_mtAB_async = Pyro4.async(remote_mt_a_b)
    # remote_mtBA_async = Pyro4.async(remote_mt_b_a)
    # remote_lmA_async = Pyro4.async(remote_lm_a)
    # remote_lmB_async = Pyro4.async(remote_lm_b)
    #
    # # asynchronously synchronize remotes
    # r0 = remote_mtAB_async.init(model_optionsAB)
    # r1 = remote_mtBA_async.init(model_optionsBA)
    # r2 = remote_lmA_async.init(language_models[0], worddicts_r[0])
    # r3 = remote_lmB_async.init(language_models[1], worddicts_r[1])
    # # synchronize
    # for x in [r0, r1, r2, r3]:
    #     _ = x.value

    print 'Remotes should be initilized'

    print 'Optimization'

    best_p = [None for _ in range(2)]
    bad_counter = 0
    # uidx = [0 for _ in range(2)] # TODO they should not be different in different models, use just one
    estop = False
    history_errs = [[] for _ in range(2)]

    for idx, model_options in zip(range(2), [model_options_a_b, model_options_b_a]):
        # reload history
        if model_options['reload_'] and os.path.exists(model_options['saveto']):
            rmodel = numpy.load(model_options['saveto'])
            history_errs[idx] = list(rmodel['history_errs'])
            # modify saveto so as not to overwrite original model
            #            model_options['saveto']=os.path.join(LOCALMODELDIR, basename(model_options['saveto']))
            # if 'uidx' in rmodel:
            #     uidx[idx] = rmodel['uidx']

            # save model options
            #        json.dump(model_options, open('%s.json' % model_options['saveto'], 'wb'), indent=2)

    if valid_freq == -1:
        valid_freq = len(training[0]) / batch_size
    if save_freq == -1:
        save_freq = len(training[0]) / batch_size
    if sample_freq == -1:
        sample_freq = len(training[0]) / batch_size

    valid_err = None

    last_disp_samples = 0
    ud_start = time.time()
    p_validation = None
    k = 2
    alpha = 0.05
    learning_rate_small = 0.0002 / batch_size  # gamma_1,t in paper, scaled by batch_size
    learning_rate_big = 0.02 / batch_size  # gamma_2,t in paper, scaled by batch_size
    for eidx in xrange(max_epochs):
        n_samples = 0

        for data_type, data in training:

            if data_type == 'mt':

                print 'training on bitext'

                for (x, y), model_options, _remote_mt in zip(data, 
                                                             [model_options_a_b, model_options_b_a],
                                                             [remote_mt_a_b,     remote_mt_b_a    ]):

                    # ensure consistency in number of factors
                    if len(x) and len(x[0]) and len(x[0][0]) != model_options['factors']:
                        sys.stderr.write(
                            'Error: mismatch between number of factors in settings ({0}), '
                            'and number in training corpus ({1})\n'.format(
                                model_options['factors'], len(x[0][0])))
                        sys.exit(1)

                    # n_samples += len(x)  # TODO double count??
                    # last_disp_samples += len(x)
                    # # uidx += 1

                    # TODO: training each system in parallel!

                    x_prep, x_mask, y_prep, y_mask = prepare_data(x, y, maxlen=maxlen)

                    _remote_mt.set_noise_val(1.)

                    if x_prep is None:
                        print 'Minibatch with zero sample under length ', maxlen
                        # uidx -= 1
                        continue

                    cost = _remote_mt.x_f_grad_shared(x_prep, x_mask, y_prep, y_mask)

                    # check for bad numbers, usually we remove non-finite elements
                    # and continue training - but not done here
                    if numpy.isnan(cost) or numpy.isinf(cost):
                        print 'NaN detected'
                        return 1., 1., 1.

                    # do the update on parameters
                    _remote_mt.x_f_update(lrate)

            elif data_type == 'mono-a':
                print '#'*40, 'training the a -> b -> a loop.'
                ret = monolingual_train([remote_mt_a_b, remote_mt_b_a], 
                                        remote_lm_b, data, trng, k, maxlen,
                                        [worddicts_r_a_b, worddicts_r_b_a], 
                                        [worddicts_a_b,   worddicts_b_a], 
                                        alpha, learning_rate_big,
                                        learning_rate_small)
            elif data_type == 'mono-b':
                print '#'*40, 'training the b -> a -> b loop.'
                ret = monolingual_train([remote_mt_b_a, remote_mt_a_b], 
                                        remote_lm_a, data, trng, k, maxlen,
                                        [worddicts_r_b_a, worddicts_r_a_b], 
                                        [worddicts_b_a,   worddicts_a_b], 
                                        alpha, learning_rate_big,
                                        learning_rate_small)
            else:
                raise Exception('This should be unreachable. How did you get here.')

    return valid_err

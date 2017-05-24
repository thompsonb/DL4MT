#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Build a neural machine translation model with soft attention
'''

import argparse
import os
import sys

import Pyro4
from theano.tensor.shared_randomstreams import RandomStreams

from data_iterator import TextIterator
from domain_interpolation_data_iterator import DomainInterpolatorTextIterator
from nmt import prepare_data
from pyro_utils import setup_remotes, get_random_key, get_unused_port
from util import load_dict

gpu_id = 2
profile = False


def train(**kwargs):
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    remote_lm_script = os.path.join(current_script_dir, 'lm_remote.py')

    pyro_key = get_random_key()
    pyro_port = get_unused_port()

    kwargs.update({'pyro_name_target_lm': 'target_lm',
                   'pyro_name_source_lm': 'source_lm',
                   'pyro_key': pyro_key,
                   'pyro_port': pyro_port})

    with setup_remotes(remote_metadata_list=[dict(script=remote_lm_script, name='target_lm', gpu_id=gpu_id),
                                             dict(script=remote_lm_script, name='source_lm', gpu_id=gpu_id), ],
                       pyro_port=pyro_port,
                       pyro_key=pyro_key):
        train2(**kwargs)


def train2(dim_word=100,  # word vector dimensionality
           dim=1000,  # the number of LSTM units
           factors=1,  # input factors
           dim_per_factor=None,
           # list of word vector dimensionalities (one per factor): [250,200,50] for total dimensionality of 500
           encoder='gru',
           decoder='gru_cond',
           patience=10,  # early stopping patience
           max_epochs=5000,
           finish_after=10000000,  # finish after this many updates
           dispFreq=100,
           decay_c=0.,  # L2 regularization penalty
           map_decay_c=0.,  # L2 regularization penalty towards original weights
           alpha_c=0.,  # alignment regularization
           clip_c=-1.,  # gradient clipping threshold
           lrate=0.01,  # learning rate
           n_words_src=None,  # source vocabulary size
           n_words=None,  # target vocabulary size
           maxlen=100,  # maximum length of the description
           optimizer='rmsprop',
           batch_size=16,
           valid_batch_size=16,
           saveto='model.npz',
           validFreq=1000,
           saveFreq=1000,  # save the parameters after every saveFreq updates
           sampleFreq=100,  # generate some samples after every sampleFreq
           datasets=['/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
                     '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],
           valid_datasets=['../data/dev/newstest2011.en.tok',
                           '../data/dev/newstest2011.fr.tok'],
           dictionaries=['/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
                         '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
           use_dropout=False,
           dropout_embedding=0.2,  # dropout for input embeddings (0: no dropout)
           dropout_hidden=0.5,  # dropout for hidden layers (0: no dropout)
           dropout_source=0,  # dropout source words (0: no dropout)
           dropout_target=0,  # dropout target words (0: no dropout)
           reload_=False,
           overwrite=False,
           external_validation_script=None,
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
           pyro_name_source_lm=None,  # if None, will import instead of assuming a server is running
           pyro_name_target_lm=None,  # if None, will import instead of assuming a server is running
           ):
    # Model options
    model_options = locals().copy()

    # I can't come up with a reason that trng must be shared... (I use a different seed)
    trng = RandomStreams(hash(__file__) % 4294967294)

    if model_options['dim_per_factor'] is None:
        if factors == 1:
            model_options['dim_per_factor'] = [model_options['dim_word']]
        else:
            sys.stderr.write('Error: if using factored input, you must specify \'dim_per_factor\'\n')
            sys.exit(1)

    assert (len(dictionaries) == factors + 1)  # one dictionary per source factor + 1 for target factor
    assert (len(model_options['dim_per_factor']) == factors)  # each factor embedding has its own dimensionality
    assert (sum(model_options['dim_per_factor']) == model_options[
        'dim_word'])  # dimensionality of factor embeddings sums up to total dimensionality of input embedding vector

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        worddicts[ii] = load_dict(dd)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    if n_words_src is None:
        n_words_src = len(worddicts[0])
        model_options['n_words_src'] = n_words_src
    if n_words is None:
        n_words = len(worddicts[1])
        model_options['n_words'] = n_words

    print 'Loading data'
    domain_interpolation_cur = None
    if use_domain_interpolation:
        print 'Using domain interpolation with initial ratio %s, increase rate %s' % (
        domain_interpolation_min, domain_interpolation_inc)
        domain_interpolation_cur = domain_interpolation_min
        train = DomainInterpolatorTextIterator(datasets[0], datasets[1],
                                               dictionaries[:-1], dictionaries[1],
                                               n_words_source=n_words_src, n_words_target=n_words,
                                               batch_size=batch_size,
                                               maxlen=maxlen,
                                               shuffle_each_epoch=shuffle_each_epoch,
                                               sort_by_length=sort_by_length,
                                               indomain_source=domain_interpolation_indomain_datasets[0],
                                               indomain_target=domain_interpolation_indomain_datasets[1],
                                               interpolation_rate=domain_interpolation_cur,
                                               maxibatch_size=maxibatch_size)
    else:
        train = TextIterator(datasets[0], datasets[1],
                             dictionaries[:-1], dictionaries[-1],
                             n_words_source=n_words_src, n_words_target=n_words,
                             batch_size=batch_size,
                             maxlen=maxlen,
                             skip_empty=True,
                             shuffle_each_epoch=shuffle_each_epoch,
                             sort_by_length=sort_by_length,
                             maxibatch_size=maxibatch_size)

    if valid_datasets and validFreq:
        valid = TextIterator(valid_datasets[0], valid_datasets[1],
                             dictionaries[:-1], dictionaries[-1],
                             n_words_source=n_words_src, n_words_target=n_words,
                             batch_size=valid_batch_size,
                             maxlen=maxlen)
    else:
        valid = None

    if pyro_name_source_lm:
        # In order to transfer numpy objects across the network, must use pickle as Pyro Serializer.
        # Also requires various environment flags (PYRO_SERIALIZERS_ACCEPTED, PYRO_SERIALIZER)
        #   for both name server and server.
        Pyro4.config.SERIALIZER = 'pickle'
        print 'Initilizing remote server'
        Pyro4.config.NS_PORT = pyro_port
        remote_target_lm = Pyro4.Proxy("PYRONAME:{0}".format(pyro_name_target_lm))
        remote_target_lm._pyroHmacKey = pyro_key
        remote_source_lm = Pyro4.Proxy("PYRONAME:{0}".format(pyro_name_source_lm))
        remote_source_lm._pyroHmacKey = pyro_key
    else:  # better for IDE
        print 'Importing server code (not running remotely)'
        from lm_remote import RemoteLM
        remote_source_lm = RemoteLM()
        remote_target_lm = RemoteLM()

    # trained ro arpa model from wmt16-scriptsLL/sample/data/corpus.ro, using cmd:
    # kenlm/build/bin/lmplz -o 5 <text >text.arpa
    remote_source_lm.init('', worddicts_r[0]) # TODO: Add language model for both the source language and target
    remote_target_lm.init('', worddicts_r[1])

    print 'Optimization'
    ctr = 0
    sentences_source, sentences_target = [], []
    for x, y in train:  # (source, target)

        sentences_source.append(' '.join([worddicts_r[0][x_i[0][0]] for x_i in x]))
        sentences_target.append(' '.join([worddicts_r[1][y_i] for y_i in y[0]]))

        # ensure consistency in number of factors
        if len(x) and len(x[0]) and len(x[0][0]) != factors:
            sys.stderr.write(
                'Error: mismatch between number of factors in settings ({0}), and number in training corpus ({1})\n'.format(
                    factors, len(x[0][0])))
            sys.exit(1)

        x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen)

        source_scores = remote_source_lm.score(x)
        print 'SOURCE SCORES:', source_scores

        target_scores = remote_target_lm.score(y)
        print 'TARGET SCORES:', target_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    data = parser.add_argument_group('data sets; model loading and saving')
    data.add_argument('--datasets', type=str, required=True, metavar='PATH', nargs=2,
                      help="parallel training corpus (source and target)")
    data.add_argument('--dictionaries', type=str, required=True, metavar='PATH', nargs="+",
                      help="network vocabularies (one per source factor, plus target vocabulary)")
    data.add_argument('--model', type=str, default='model.npz', metavar='PATH', dest='saveto',
                      help="model file name (default: %(default)s)")
    data.add_argument('--saveFreq', type=int, default=30000, metavar='INT',
                      help="save frequency (default: %(default)s)")
    data.add_argument('--reload', action='store_true', dest='reload_',
                      help="load existing model (if '--model' points to existing model)")
    data.add_argument('--overwrite', action='store_true',
                      help="write all models to same file")

    network = parser.add_argument_group('network parameters')
    network.add_argument('--dim_word', type=int, default=512, metavar='INT',
                         help="embedding layer size (default: %(default)s)")
    network.add_argument('--dim', type=int, default=1000, metavar='INT',
                         help="hidden layer size (default: %(default)s)")
    network.add_argument('--n_words_src', type=int, default=None, metavar='INT',
                         help="source vocabulary size (default: %(default)s)")
    network.add_argument('--n_words', type=int, default=None, metavar='INT',
                         help="target vocabulary size (default: %(default)s)")

    network.add_argument('--factors', type=int, default=1, metavar='INT',
                         help="number of input factors (default: %(default)s)")
    network.add_argument('--dim_per_factor', type=int, default=None, nargs='+', metavar='INT',
                         help="list of word vector dimensionalities (one per factor): '--dim_per_factor 250 200 50' for total dimensionality of 500 (default: %(default)s)")
    network.add_argument('--use_dropout', action="store_true",
                         help="use dropout layer (default: %(default)s)")
    network.add_argument('--dropout_embedding', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for input embeddings (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_hidden', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for hidden layer (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_source', type=float, default=0, metavar="FLOAT",
                         help="dropout source words (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_target', type=float, default=0, metavar="FLOAT",
                         help="dropout target words (0: no dropout) (default: %(default)s)")

    training = parser.add_argument_group('training parameters')
    training.add_argument('--maxlen', type=int, default=100, metavar='INT',
                          help="maximum sequence length (default: %(default)s)")
    training.add_argument('--optimizer', type=str, default="adam",
                          choices=['adam', 'adadelta', 'rmsprop', 'sgd'],
                          help="optimizer (default: %(default)s)")
    training.add_argument('--batch_size', type=int, default=80, metavar='INT',
                          help="minibatch size (default: %(default)s)")
    training.add_argument('--max_epochs', type=int, default=5000, metavar='INT',
                          help="maximum number of epochs (default: %(default)s)")
    training.add_argument('--finish_after', type=int, default=10000000, metavar='INT',
                          help="maximum number of updates (minibatches) (default: %(default)s)")
    training.add_argument('--decay_c', type=float, default=0, metavar='FLOAT',
                          help="L2 regularization penalty (default: %(default)s)")
    training.add_argument('--map_decay_c', type=float, default=0, metavar='FLOAT',
                          help="L2 regularization penalty towards original weights (default: %(default)s)")
    training.add_argument('--alpha_c', type=float, default=0, metavar='FLOAT',
                          help="alignment regularization (default: %(default)s)")
    training.add_argument('--clip_c', type=float, default=1, metavar='FLOAT',
                          help="gradient clipping threshold (default: %(default)s)")
    training.add_argument('--lrate', type=float, default=0.0001, metavar='FLOAT',
                          help="learning rate (default: %(default)s)")
    training.add_argument('--no_shuffle', action="store_false", dest="shuffle_each_epoch",
                          help="disable shuffling of training data (for each epoch)")
    training.add_argument('--no_sort_by_length', action="store_false", dest="sort_by_length",
                          help='do not sort sentences in maxibatch by length')
    training.add_argument('--maxibatch_size', type=int, default=20, metavar='INT',
                          help='size of maxibatch (number of minibatches that are sorted by length) (default: %(default)s)')

    finetune = training.add_mutually_exclusive_group()
    finetune.add_argument('--finetune', action="store_true",
                          help="train with fixed embedding layer")
    finetune.add_argument('--finetune_only_last', action="store_true",
                          help="train with all layers except output layer fixed")

    validation = parser.add_argument_group('validation parameters')
    validation.add_argument('--valid_datasets', type=str, default=None, metavar='PATH', nargs=2,
                            help="parallel validation corpus (source and target) (default: %(default)s)")
    validation.add_argument('--valid_batch_size', type=int, default=80, metavar='INT',
                            help="validation minibatch size (default: %(default)s)")
    validation.add_argument('--validFreq', type=int, default=10000, metavar='INT',
                            help="validation frequency (default: %(default)s)")
    validation.add_argument('--patience', type=int, default=10, metavar='INT',
                            help="early stopping patience (default: %(default)s)")
    validation.add_argument('--external_validation_script', type=str, default=None, metavar='PATH',
                            help="location of validation script (to run your favorite metric for validation) (default: %(default)s)")

    display = parser.add_argument_group('display parameters')
    display.add_argument('--dispFreq', type=int, default=1000, metavar='INT',
                         help="display loss after INT updates (default: %(default)s)")
    display.add_argument('--sampleFreq', type=int, default=10000, metavar='INT',
                         help="display some samples after INT updates (default: %(default)s)")

    args = parser.parse_args()

    train(**vars(args))  # use train2(...) instead to bypass Pyro

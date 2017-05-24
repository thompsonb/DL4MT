#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a neural machine translation model with soft attention
"""

import argparse
import copy
import json
import os
import sys
import time
from collections import OrderedDict
from subprocess import Popen

import ipdb
import numpy
import theano
import theano.tensor as tensor

from data_iterator import TextIterator
from domain_interpolation_data_iterator import DomainInterpolatorTextIterator
from nmt_utils import init_params, build_model, build_sampler, prepare_data, gen_sample, pred_probs
from theano_util import load_params, init_theano_params, itemlist, unzip_from_theano, zip_to_theano
from util import load_dict

profile = False


def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          factors=1, # input factors
          dim_per_factor=None, # list of word vector dimensionalities (one per factor): [250,200,50] for total dimensionality of 500
          encoder='gru',
          decoder='gru_cond',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          map_decay_c=0., # L2 regularization penalty towards original weights
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
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=('/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
                    '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'),
          valid_datasets=('../data/dev/newstest2011.en.tok',
                          '../data/dev/newstest2011.fr.tok'),
          dictionaries=('/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
                        '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'),
          use_dropout=False,
          dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
          dropout_hidden=0.5, # dropout for hidden layers (0: no dropout)
          dropout_source=0, # dropout source words (0: no dropout)
          dropout_target=0, # dropout target words (0: no dropout)
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
          domain_interpolation_indomain_datasets=('indomain.en', 'indomain.fr'),
          maxibatch_size=20, #How many minibatches to load at one time
          model_version=0.1, #store version used for training for compatibility
    ):

    # Model options
    model_options = locals().copy()

    if model_options['dim_per_factor'] == None:
        if factors == 1:
            model_options['dim_per_factor'] = [model_options['dim_word']]
        else:
            sys.stderr.write('Error: if using factored input, you must specify \'dim_per_factor\'\n')
            sys.exit(1)

    assert(len(dictionaries) == factors + 1) # one dictionary per source factor + 1 for target factor
    assert(len(model_options['dim_per_factor']) == factors) # each factor embedding has its own dimensionality
    assert(sum(model_options['dim_per_factor']) == model_options['dim_word']) # dimensionality of factor embeddings sums up to total dimensionality of input embedding vector

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

    print('Loading data')
    domain_interpolation_cur = None
    if use_domain_interpolation:
        print('Using domain interpolation with initial ratio %s, increase rate %s' % (domain_interpolation_min, domain_interpolation_inc))
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

    comp_start = time.time()

    print('Building model')
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        print('Reloading model parameters')
        params = load_params(saveto, params)

    tparams = init_theano_params(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)

    inps = [x, x_mask, y, y_mask]

    if validFreq or sampleFreq:
        print('Building sampler')
        f_init, f_next = build_sampler(tparams, model_options, use_noise, trng)

    # before any regularizer
    print('Building f_log_probs...',)
    f_log_probs = theano.function(inps, cost, profile=profile)
    print('Done')

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # apply L2 regularisation to loaded model (map training)
    if map_decay_c > 0:
        map_decay_c = theano.shared(numpy.float32(map_decay_c), name="map_decay_c")
        weight_map_decay = 0.
        for kk, vv in tparams.iteritems():
            init_value = theano.shared(vv.get_value(), name= kk + "_init")
            weight_map_decay += ((vv -init_value) ** 2).sum()
        weight_map_decay *= map_decay_c
        cost += weight_map_decay

    # allow finetuning with fixed embeddings
    if finetune:
        updated_params = OrderedDict([(key,value) for (key,value) in tparams.iteritems() if not key.startswith('Wemb')])
    else:
        updated_params = tparams

    # allow finetuning of only last layer (becomes a linear model training problem)
    if finetune_only_last:
        updated_params = OrderedDict([(key,value) for (key,value) in tparams.iteritems() if key in ['ff_logit_W', 'ff_logit_b']])
    else:
        updated_params = tparams

    print('Computing gradient...',)
    grads = tensor.grad(cost, wrt=itemlist(updated_params))
    print('Done')

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')

    print('Building optimizers...',)
    f_grad_shared, f_update = eval(optimizer)(lr, updated_params, grads, inps, cost, profile=profile)
    print('Done')

    print('Total compilation time: {0:.1f}s'.format(time.time() - comp_start))

    print('Optimization')

    best_p = None
    bad_counter = 0
    uidx = 0
    estop = False
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        rmodel = numpy.load(saveto)
        history_errs = list(rmodel['history_errs'])
        if 'uidx' in rmodel:
            uidx = rmodel['uidx']

    # save model options
    json.dump(model_options, open('%s.json' % saveto, 'wb'), indent=2)

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    valid_err = None

    last_disp_samples = 0
    ud_start = time.time()
    p_validation = None
    for eidx in xrange(max_epochs):
        n_samples = 0

        for x, y in train:
            n_samples += len(x)
            last_disp_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            # ensure consistency in number of factors
            if len(x) and len(x[0]) and len(x[0][0]) != factors:
                sys.stderr.write('Error: mismatch between number of factors in settings ({0}), and number in training corpus ({1})\n'.format(factors, len(x[0][0])))
                sys.exit(1)

            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen)  # n_words_src=n_words_src, n_words=n_words) # TODO: why unused??

            if x is None:
                print('Minibatch with zero sample under length ', maxlen)
                uidx -= 1
                continue

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, y, y_mask)

            # do the update on parameters
            f_update(lrate)

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print('NaN detected')
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                ud = time.time() - ud_start
                wps = (last_disp_samples) / float(ud)
                print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud, "{0:.2f} sentences/s".format(wps))
                ud_start = time.time()
                last_disp_samples = 0

            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
            if numpy.mod(uidx, saveFreq) == 0:
                print('Saving the best model...',)
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip_from_theano(tparams)
                numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **params)
                print('Done')

                # save with uidx
                if not overwrite:
                    print('Saving the model at iteration {}...'.format(uidx),)
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    numpy.savez(saveto_uidx, history_errs=history_errs,
                                uidx=uidx, **unzip_from_theano(tparams))
                    print('Done')

            # generate some samples with the model and display them
            if sampleFreq and numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[2])):
                    stochastic = True
                    x_current = x[:, :, jj][:, :, None]

                    # remove padding
                    x_current = x_current[:,:x_mask[:, jj].sum(),:]

                    sample, score, sample_word_probs, alignment, hyp_graph = gen_sample([f_init], [f_next],
                                                                                        x_current,
                                                                                        trng=trng, k=1,
                                                                                        maxlen=30,
                                                                                        stochastic=stochastic,
                                                                                        argmax=False,
                                                                                        suppress_unk=False,
                                                                                        return_hyp_graph=False)
                    print('Source ', jj, ': ',)
                    for pos in range(x.shape[1]):
                        if x[0, pos, jj] == 0:
                            break
                        for factor in range(factors):
                            vv = x[factor, pos, jj]
                            if vv in worddicts_r[factor]:
                                sys.stdout.write(worddicts_r[factor][vv])
                            else:
                                sys.stdout.write('UNK')
                            if factor+1 < factors:
                                sys.stdout.write('|')
                            else:
                                sys.stdout.write(' ')
                    print()
                    print('Truth ', jj, ' : ',)
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[-1]:
                            print(worddicts_r[-1][vv],)
                        else:
                            print('UNK',)
                    print()
                    print('Sample ', jj, ': ',)
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r[-1]:
                            print(worddicts_r[-1][vv],)
                        else:
                            print('UNK',)
                    print()

            # validate model on validation set and early stop if necessary
            if valid and validFreq and numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs, alignment = pred_probs(f_log_probs, prepare_data,
                                                   model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip_from_theano(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        if use_domain_interpolation and (domain_interpolation_cur < 1.0):
                            domain_interpolation_cur = min(domain_interpolation_cur + domain_interpolation_inc, 1.0)
                            print('No progress on the validation set, increasing domain interpolation rate to %s and resuming from best params' % domain_interpolation_cur)
                            train.adjust_domain_interpolation_rate(domain_interpolation_cur)
                            if best_p is not None:
                                zip_to_theano(best_p, tparams)
                            bad_counter = 0
                        else:
                            print('Early Stop!')
                            estop = True
                            break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                print('Valid ', valid_err)

                if external_validation_script:
                    print("Calling external validation script")
                    if p_validation is not None and p_validation.poll() is None:
                        print("Waiting for previous validation run to finish")
                        print("If this takes too long, consider increasing validation interval, reducing validation set size, or speeding up validation by using multiple processes")
                        valid_wait_start = time.time()
                        p_validation.wait()
                        print("Waited for {0:.1f} seconds".format(time.time()-valid_wait_start))
                    print('Saving  model...',)
                    params = unzip_from_theano(tparams)
                    numpy.savez(saveto +'.dev', history_errs=history_errs, uidx=uidx, **params)
                    json.dump(model_options, open('%s.dev.npz.json' % saveto, 'wb'), indent=2)
                    print('Done')
                    p_validation = Popen([external_validation_script])

            # finish after this many updates
            if uidx >= finish_after:
                print('Finishing after %d iterations!' % uidx)
                estop = True
                break

        print('Seen %d samples' % n_samples)

        if estop:
            break

    if best_p is not None:
        zip_to_theano(best_p, tparams)

    if valid:
        use_noise.set_value(0.)
        valid_errs, alignment = pred_probs(f_log_probs, prepare_data,
                                           model_options, valid)
        valid_err = valid_errs.mean()

        print('Valid ', valid_err)

    if best_p is not None:
        params = copy.copy(best_p)
    else:
        params = unzip_from_theano(tparams)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                uidx=uidx,
                **params)

    return valid_err


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
    data.add_argument('--reload', action='store_true',  dest='reload_',
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
    #network.add_argument('--encoder', type=str, default='gru',
                         #choices=['gru'],
                         #help='encoder recurrent layer')
    #network.add_argument('--decoder', type=str, default='gru_cond',
                         #choices=['gru_cond'],
                         #help='decoder recurrent layer')

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

    train(**vars(args))

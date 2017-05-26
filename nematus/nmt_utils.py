#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a neural machine translation model with soft attention
"""

import copy
import sys
from collections import OrderedDict

import ipdb
import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from initializers import norm_weight
from layers import get_layer_param, shared_dropout_layer, get_layer_constr
from theano_util import concatenate, embedding_name
from alignment_util import get_alignments

profile = False

# Nematus relies on numpy.log(-numpy.inf) for suppressing unknowns
# make sure numpy will not raise an exception because of nan
numpy.seterr(divide='warn', over='warn', under='ignore', invalid='warn')

# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    n_factors = len(seqs_x[0][0])
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((n_factors, maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:, :lengths_x[idx], idx] = zip(*s_x)
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask


# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    for factor in range(options['factors']):
        params[embedding_name(factor)] = norm_weight(options['n_words_src'], options['dim_per_factor'][factor])

    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

    # encoder: bidirectional RNN
    params = get_layer_param(options['encoder'])(options, params,
                                                 prefix='encoder',
                                                 nin=options['dim_word'],
                                                 dim=options['dim'])
    params = get_layer_param(options['encoder'])(options, params,
                                                 prefix='encoder_r',
                                                 nin=options['dim_word'],
                                                 dim=options['dim'])
    ctxdim = 2 * options['dim']

    # init_state, init_cell
    params = get_layer_param('ff')(options, params, prefix='ff_state',
                                   nin=ctxdim, nout=options['dim'])
    # decoder
    params = get_layer_param(options['decoder'])(options, params,
                                                 prefix='decoder',
                                                 nin=options['dim_word'],
                                                 dim=options['dim'],
                                                 dimctx=ctxdim)
    # readout
    params = get_layer_param('ff')(options, params, prefix='ff_logit_lstm',
                                   nin=options['dim'], nout=options['dim_word'],
                                   ortho=False)
    params = get_layer_param('ff')(options, params, prefix='ff_logit_prev',
                                   nin=options['dim_word'],
                                   nout=options['dim_word'], ortho=False)
    params = get_layer_param('ff')(options, params, prefix='ff_logit_ctx',
                                   nin=ctxdim, nout=options['dim_word'],
                                   ortho=False)
    params = get_layer_param('ff')(options, params, prefix='ff_logit',
                                   nin=options['dim_word'],
                                   nout=options['n_words'])

    return params


# bidirectional RNN encoder: take input x (optionally with mask), and produce sequence of context vectors (ctx)
def _build_encoder(tparams, options, trng, use_noise, x_mask=None, sampling=False):

    x = tensor.tensor3('x', dtype='int64')
    x.tag.test_value = (numpy.random.rand(1, 5, 10)*100).astype('int64')

    # for the backward rnn, we just need to invert x
    xr = x[:,::-1]
    if x_mask is None:
        xr_mask = None
    else:
        xr_mask = x_mask[::-1]

    n_timesteps = x.shape[1]
    n_samples = x.shape[2]

    if options['use_dropout']:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_source = 1-options['dropout_source']
        if sampling:
            if options['model_version'] < 0.1:
                rec_dropout = theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32'))
                rec_dropout_r = theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32'))
                emb_dropout = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
                emb_dropout_r = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
                source_dropout = theano.shared(numpy.float32(retain_probability_source))
            else:
                rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
                rec_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
                emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
                emb_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
                source_dropout = theano.shared(numpy.float32(1.))
        else:
            if options['model_version'] < 0.1:
                scaled = False
            else:
                scaled = True
            rec_dropout = shared_dropout_layer((2, n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
            rec_dropout_r = shared_dropout_layer((2, n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
            emb_dropout = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
            emb_dropout_r = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
            source_dropout = shared_dropout_layer((n_timesteps, n_samples, 1), use_noise, trng, retain_probability_source, scaled)
            source_dropout = tensor.tile(source_dropout, (1,1,options['dim_word']))
    else:
        rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        rec_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))

    # word embedding for forward rnn (source)
    emb = []
    for factor in range(options['factors']):
        emb.append(tparams[embedding_name(factor)][x[factor].flatten()])
    emb = concatenate(emb, axis=1)
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    if options['use_dropout']:
        emb *= source_dropout

    proj = get_layer_constr(options['encoder'])(tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask,
                                            emb_dropout=emb_dropout, 
                                            rec_dropout=rec_dropout,
                                            profile=profile)

    # word embedding for backward rnn (source)
    embr = []
    for factor in range(options['factors']):
        embr.append(tparams[embedding_name(factor)][xr[factor].flatten()])
    embr = concatenate(embr, axis=1)
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    if options['use_dropout']:
        if sampling:
            embr *= source_dropout
        else:
            # we drop out the same words in both directions
            embr *= source_dropout[::-1]

    projr = get_layer_constr(options['encoder'])(tparams, embr, options,
                                             prefix='encoder_r',
                                             mask=xr_mask,
                                             emb_dropout=emb_dropout_r,
                                             rec_dropout=rec_dropout_r,
                                             profile=profile)

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    return x, ctx


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    x_mask = tensor.matrix('x_mask', dtype='float32')
    x_mask.tag.test_value = numpy.ones(shape=(5, 10)).astype('float32')
    y = tensor.matrix('y', dtype='int64')
    y.tag.test_value = (numpy.random.rand(8, 10)*100).astype('int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')
    y_mask.tag.test_value = numpy.ones(shape=(8, 10)).astype('float32')

    x, ctx = _build_encoder(tparams, options, trng, use_noise, x_mask, sampling=False)
    n_samples = x.shape[2]
    n_timesteps_trg = y.shape[0]

    if options['use_dropout']:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_target = 1-options['dropout_target']
        if options['model_version'] < 0.1:
            scaled = False
        else:
            scaled = True
        rec_dropout_d = shared_dropout_layer((5, n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
        emb_dropout_d = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
        ctx_dropout_d = shared_dropout_layer((4, n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden, scaled)
        target_dropout = shared_dropout_layer((n_timesteps_trg, n_samples, 1), use_noise, trng, retain_probability_target, scaled)
        target_dropout = tensor.tile(target_dropout, (1,1,options['dim_word']))
    else:
        rec_dropout_d = theano.shared(numpy.array([1.]*5, dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([1.]*2, dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([1.]*4, dtype='float32'))

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    if options['use_dropout']:
        ctx_mean *= shared_dropout_layer((n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden, scaled)

    # initial decoder state
    init_state = get_layer_constr('ff')(tparams, ctx_mean, options,
                                        prefix='ff_state', activ='tanh')

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])

    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    if options['use_dropout']:
        emb *= target_dropout

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer_constr(options['decoder'])(tparams, emb, options,
                                                prefix='decoder',
                                                mask=y_mask, context=ctx,
                                                context_mask=x_mask,
                                                one_step=False,
                                                init_state=init_state,
                                                emb_dropout=emb_dropout_d,
                                                ctx_dropout=ctx_dropout_d,
                                                rec_dropout=rec_dropout_d,
                                                profile=profile)
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    if options['use_dropout']:
        proj_h *= shared_dropout_layer((n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
        emb *= shared_dropout_layer((n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
        ctxs *= shared_dropout_layer((n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden, scaled)

    # weights (alignment matrix) #####LIUCAN: this is where the attention vector is.
    opt_ret['dec_alphas'] = proj[2]

    # compute word probabilities
    logit_lstm = get_layer_constr('ff')(tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer_constr('ff')(tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer_constr('ff')(tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

    if options['use_dropout']:
        logit *= shared_dropout_layer((n_samples, options['dim_word']), use_noise, trng, retain_probability_hidden, scaled)

    logit = get_layer_constr('ff')(tparams, logit, options,
                                   prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    per_sent_neg_log_prob = -tensor.log(probs.flatten()[y_flat_idx])
    per_sent_neg_log_prob = per_sent_neg_log_prob.reshape([y.shape[0], y.shape[1]])
    per_sent_neg_log_prob = (per_sent_neg_log_prob * y_mask).sum(0)  # note: y_mask is float, but only stores 0. or 1.

    #print "Print out in build_model()"
    #print opt_ret
    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, per_sent_neg_log_prob

# build a batched sampler
def build_sampler(tparams, options, use_noise, trng, return_alignment=False):

    x_mask = tensor.matrix('x_mask', dtype='float32')
    x_mask.tag.test_value = numpy.ones(shape=(5, 10)).astype('float32')

    if options['use_dropout'] and options['model_version'] < 0.1:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        # retain_probability_source = 1-options['dropout_source']  # todo: should this be used??
        retain_probability_target = 1-options['dropout_target']
        rec_dropout_d = theano.shared(numpy.array([retain_probability_hidden]*5, dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([retain_probability_hidden]*4, dtype='float32'))
        target_dropout = theano.shared(numpy.float32(retain_probability_target))
    else:
        rec_dropout_d = theano.shared(numpy.array([1.]*5, dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([1.]*2, dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([1.]*4, dtype='float32'))

    x, ctx = _build_encoder(tparams, options, trng, use_noise, x_mask=x_mask, sampling=True)

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)

    if options['use_dropout'] and options['model_version'] < 0.1:
        ctx_mean *= retain_probability_hidden

    init_state = get_layer_constr('ff')(tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print >>sys.stderr, 'Building f_init...',
    outs = [init_state, ctx]
    f_init = theano.function([x, x_mask], outs, name='f_init', profile=profile)
    print >>sys.stderr, 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    if options['use_dropout'] and options['model_version'] < 0.1:
        emb *= target_dropout

    # apply one step of conditional gru with attention
    proj = get_layer_constr(options['decoder'])(tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            context_mask=x_mask,
                                            one_step=True,
                                            init_state=init_state,
                                            emb_dropout=emb_dropout_d,
                                            ctx_dropout=ctx_dropout_d,
                                            rec_dropout=rec_dropout_d,
                                            profile=profile)
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]

    # alignment matrix (attention model)
    dec_alphas = proj[2]

    if options['use_dropout'] and options['model_version'] < 0.1:
        next_state_up = next_state * retain_probability_hidden
        emb *= retain_probability_emb
        ctxs *= retain_probability_hidden
    else:
        next_state_up = next_state

    logit_lstm = get_layer_constr('ff')(tparams, next_state_up, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer_constr('ff')(tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer_constr('ff')(tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

    if options['use_dropout'] and options['model_version'] < 0.1:
        logit *= retain_probability_hidden

    logit = get_layer_constr('ff')(tparams, logit, options,
                              prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print >>sys.stderr, 'Building f_next..',
    inps = [y, ctx, init_state, x_mask]
    outs = [next_probs, next_sample, next_state]

    if return_alignment:
        outs.append(dec_alphas)

    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print >>sys.stderr, 'Done'

    return f_init, f_next


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(f_init, f_next, x, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False, return_alignment=False, suppress_unk=False,
               return_hyp_graph=False):
    """
    :param f_init: *list* of f_init functions. Each: state0, ctx0 = f_init(x)
    :param f_next: *list* of f_next functions. Each: next_prob, next_word, next_state = f_next(word, ctx, state)
    :param x: a [BATCHED?] sequence of word ids followed by 0 (0 = eos id)
    :param trng: theano RandomStreams
    :param k: beam width
    :param maxlen: max length of a sentences
    :param stochastic: bool, do stochastic sampling
    :param argmax: bool, something to do with argmax of highest word prob...
    :param return_alignment:
    :param suppress_unk:
    :param return_hyp_graph:
    :return:
    """

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    sample_word_probs = []
    alignment = []
    hyp_graph = None
    if stochastic:
        sample_score = 0
    if return_hyp_graph:
        from hypgraph import HypGraph
        hyp_graph = HypGraph()

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    word_probs = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    if return_alignment:
        hyp_alignment = [[] for _ in xrange(live_k)]

    # for ensemble decoding, we keep track of states and probability distribution
    # for each model in the ensemble
    num_models = len(f_init)
    next_state = [None]*num_models
    ctx0 = [None]*num_models
    next_p = [None]*num_models
    dec_alphas = [None]*num_models
    # get initial state of decoder rnn and encoder context
    for i in xrange(num_models):
        ret = f_init[i](x)
        next_state[i] = ret[0]
        ctx0[i] = ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    # x is a sequence of word ids followed by 0, eos id
    for ii in xrange(maxlen):
        for i in xrange(num_models):
            ctx = numpy.tile(ctx0[i], [live_k, 1])
            inps = [next_w, ctx, next_state[i]]
            ret = f_next[i](*inps)
            # dimension of dec_alpha (k-beam-size, number-of-input-hidden-units)
            next_p[i], next_w_tmp, next_state[i] = ret[0], ret[1], ret[2]
            if return_alignment:
                dec_alphas[i] = ret[3]

            if suppress_unk:
                next_p[i][:,1] = -numpy.inf
        if stochastic:
            if argmax:
                nw = sum(next_p)[0].argmax()
            else:
                nw = next_w_tmp[0]
            sample.append(nw)
            sample_score += numpy.log(next_p[0][0, nw])
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - sum(numpy.log(next_p))
            probs = sum(next_p)/num_models
            cand_flat = cand_scores.flatten()
            probs_flat = probs.flatten()
            ranks_flat = cand_flat.argpartition(k-dead_k-1)[:(k-dead_k)]

            # averaging the attention weights accross models
            if return_alignment:
                mean_alignment = sum(dec_alphas)/num_models

            voc_size = next_p[0].shape[1]
            # index of each k-best hypothesis
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_word_probs = []
            new_hyp_states = []
            if return_alignment:
                # holds the history of attention weights for each time step for each of the surviving hypothesis
                # dimensions (live_k * target_words * source_hidden_units]
                # at each time step we append the attention weights corresponding to the current target word
                new_hyp_alignment = [[] for _ in xrange(k-dead_k)]

            # ti -> index of k-best hypothesis
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_word_probs.append(word_probs[ti] + [probs_flat[ranks_flat[idx]].tolist()])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append([copy.copy(next_state[i][ti]) for i in xrange(num_models)])
                if return_alignment:
                    # get history of attention weights for the current hypothesis
                    new_hyp_alignment[idx] = copy.copy(hyp_alignment[ti])
                    # extend the history with current attention weights
                    new_hyp_alignment[idx].append(mean_alignment[ti])


            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            word_probs = []
            if return_alignment:
                hyp_alignment = []

            # sample and sample_score hold the k-best translations and their scores
            for idx in xrange(len(new_hyp_samples)):
                if return_hyp_graph:
                    word, history = new_hyp_samples[idx][-1], new_hyp_samples[idx][:-1]
                    score = new_hyp_scores[idx]
                    word_prob = new_word_probs[idx][-1]
                    hyp_graph.add(word, history, word_prob=word_prob, cost=score)
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    sample_word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        alignment.append(new_hyp_alignment[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        hyp_alignment.append(new_hyp_alignment[idx])
            hyp_scores = numpy.array(hyp_scores)

            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = [numpy.array(state) for state in zip(*hyp_states)]

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
                sample_word_probs.append(word_probs[idx])
                if return_alignment:
                    alignment.append(hyp_alignment[idx])

    if not return_alignment:
        alignment = [None for i in range(len(sample))]

    return sample, sample_score, sample_word_probs, alignment, hyp_graph


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True, normalize=False, alignweights=False):
    probs = []
    n_done = 0

    alignments_json = []

    for x, y in iterator:
        # ensure consistency in number of factors
        if len(x[0][0]) != options['factors']:
            sys.stderr.write('Error: mismatch between number of factors in settings ({0}), and number in validation corpus ({1})\n'.format(options['factors'], len(x[0][0])))
            sys.exit(1)

        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, y)
                                            #n_words_src=options['n_words_src'],
                                            #n_words=options['n_words'])

        # in optional save weights mode.
        if alignweights:
            pprobs, attention = f_log_probs(x, x_mask, y, y_mask)
            for jdata in get_alignments(attention, x_mask, y_mask):
                alignments_json.append(jdata)
        else:
            pprobs = f_log_probs(x, x_mask, y, y_mask)

        # normalize scores according to output length
        if normalize:
            lengths = numpy.array([numpy.count_nonzero(s) for s in y_mask.T])
            pprobs /= lengths

        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % n_done

    return numpy.array(probs), alignments_json


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_par_sample(f_init, f_next, x, x_mask, k=1, maxlen=30, suppress_unk=False):
    """
    :param f_init: *list* of f_init functions. Each: state0, ctx0 = f_init(x, X_MASK)
    :param f_next: *list* of f_next functions. Each: next_prob, next_word, next_state = f_next(word, ctx, state, X_MASK)
    :param x: a BATCHED sequence of word ids, each terminated by 0 (0 = eos id)
    :param k: beam width
    :param maxlen: max length of a sentences
    :param suppress_unk:
    :return:
    """
    # k is the beam size we have
    batch_size = x.shape[2]
    sample = [[] for i in range(batch_size)]
    sample_score = [[] for i in range(batch_size)]
    sample_word_probs = [[] for i in range(batch_size)]
    live_k = [1] * batch_size
    dead_k = [0] * batch_size # num completed sentences

    hyp_samples = [[]] * batch_size * 1 # wrote 1 explictly to denote 1 live_k per sent
    word_probs = [[]] * batch_size * 1
    hyp_scores = numpy.zeros(batch_size * 1).astype('float32')
    # for ensemble decoding, we keep track of states and probability distribution
    # for each model in the ensemble
    num_models = len(f_init)
    next_state = [None]*num_models
    ctx0 = [None]*num_models # initial context
    next_ps = [None]*num_models
    # get initial state of decoder rnn and encoder context
    for i in xrange(num_models):
        ret = f_init[i](x, x_mask)
        next_state[i] = ret[0]
        ctx0[i] = ret[1]
    next_w = -1 * numpy.ones((batch_size,)).astype('int64')  # bos (beginning of sent) indicator

    # -- OK --
    # x is a sequence of word ids followed by 0, eos id
    for ii in xrange(maxlen):
        for i in xrange(num_models):
            # Encoder context does not change, just need to know how many are required
            # numpy.tile(ctx0[i], [live_k, 1]) -- prev: simply repeat context live_k times (and propagate up a dimension)
            # New: tile each context per sent, then concat them together.
            ctx = numpy.concatenate([numpy.tile(ctx0[i][:,sent_idx:sent_idx+1], [live_k_per_sent, 1]) for sent_idx, live_k_per_sent in enumerate(live_k)], axis = 1)
            mask = numpy.concatenate([numpy.tile(x_mask[:,sent_idx:sent_idx+1], [1, live_k_per_sent]) for sent_idx, live_k_per_sent in enumerate(live_k)], axis = 1)
            
            inps = [next_w, ctx, next_state[i], mask] # prepare parameters for f_next
            ret = f_next[i](*inps)  
            # dimension of dec_alpha (k-beam-size, number-of-input-hidden-units)
            next_ps[i], next_ws_tmp, next_state[i] = ret[0], ret[1], ret[2]
            if suppress_unk:
                next_ps[i][:, 1] = -numpy.inf
        # We do the same thing with the same flat structures. just our interpretations now differ!
        voc_size = next_ps[0].shape[1] # should be constant
        cand_scores = hyp_scores[:, None] - sum(numpy.log(next_ps))
        probs = sum(next_ps)/num_models
        cand_flat = cand_scores.flatten()
        probs_flat = probs.flatten()
        # OK argpartition in pieces.
        # Wait if we are argpartitioning across sent boundaries, two words can come out of a single hyp! wait that's ok though! great.
        #ranks_flat = cand_flat.argpartition(k-dead_k-1)[:(k-dead_k)] # Basically, top k-dead_k (INDICES OF)
        sent_boundaries = numpy.cumsum([0] + live_k) * voc_size # Mult by vocab size because the softmaxes are flattened too
        # start, end = start index and end index for a sentence (not inclusive on end)
        # argpartition each piece. add 'start' to it because np thinks its a new small array, so remember the start idx
        ranks_flat = numpy.concatenate([start + cand_flat[start:end].argpartition(k - dead_per_sent-1)[:(k-dead_per_sent)] 
                                          for start, end, dead_per_sent in zip(sent_boundaries[:-1], sent_boundaries[1:], dead_k)], axis = 0)
        
        # averaging the attention weights across models
        # index of each k-best hypothesis
        trans_indices = ranks_flat / voc_size # Hmm. Which element of beam did it come from
        word_indices = ranks_flat % voc_size # and what word was it. That implies the cand scores are the entire softmax...
        costs = cand_flat[ranks_flat] # Get the probs
        new_hyp_samples = [] 
        new_hyp_scores = numpy.zeros(len(ranks_flat)).astype('float32')
        new_word_probs = []
        new_hyp_states = []
        # ti -> index of k-best hypothesis
        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            # hyps/etc will proceed in order, since ranks flat goes in order of sentences.
            new_hyp_samples.append(hyp_samples[ti]+[wi]) # looks like appending the next word to the existing hypothesis, and adding that to a list of new hypotheses
            new_word_probs.append(word_probs[ti] + [probs_flat[ranks_flat[idx]].tolist()]) # Not sure, I think same thing but for probabilities. the '+' -- the second element is a list so probably still a list of word probs over the hyp
            new_hyp_scores[idx] = copy.copy(costs[idx]) # the total cost with the new prob added
            new_hyp_states.append([copy.copy(next_state[i][ti]) for i in xrange(num_models)]) # copy the state over too
        # check the finished samples
        new_live_k = [0] * batch_size
        hyp_samples = []
        hyp_scores = []
        hyp_states = []
        word_probs = []
        # sample and sample_score hold the k-best translations and their scores
        # In the flattened 'sample' array, markers between sentences will be based on the cumulative sum of live_ks
        #ipdb.set_trace()

        live_k_tmp = [k-dead_k_sent for dead_k_sent in dead_k]
        sample_sent_boundaries = numpy.cumsum(live_k_tmp)
        sent_idx = 0
        for idx in xrange(len(new_hyp_samples)):
            # Need to know which sent it came from
            while idx >= sample_sent_boundaries[sent_idx]:
                sent_idx += 1
            if new_hyp_samples[idx][-1] == 0: # If eos (End of sent)
                #ipdb.set_trace()

                sample[sent_idx].append(new_hyp_samples[idx]) # I think 'sample' are only for finished samples
                sample_score[sent_idx].append(new_hyp_scores[idx])
                sample_word_probs[sent_idx].append(new_word_probs[idx])
                dead_k[sent_idx] += 1 # finished the sent.
            else:
                new_live_k[sent_idx] += 1 # count live k's
                # Live ones are flat.
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states.append(new_hyp_states[idx])
                word_probs.append(new_word_probs[idx])
        hyp_scores = numpy.array(hyp_scores)
        live_k = new_live_k
        # Conservative break conditions...
        if sum(new_live_k) < 1:
            break
        if min(dead_k) >= k:
            break
        next_w = numpy.array([w[-1] for w in hyp_samples])
        next_state = [numpy.array(state) for state in zip(*hyp_states)]
    # dump every remaining one
    sample_sent_boundaries = numpy.cumsum(live_k)
    sent_idx = 0
    for idx in xrange(len(hyp_samples)):
        while idx >= sample_sent_boundaries[sent_idx]:
            sent_idx += 1
        #if live_k > 0: 
        sample[sent_idx].append(hyp_samples[idx])
        sample_score[sent_idx].append(hyp_scores[idx])
        sample_word_probs[sent_idx].append(word_probs[idx])
    return sample, sample_score, sample_word_probs

# My notes: doing it in par:
# All of this stuff, topk etc, is done in numpy. Only f_next() etc are done in theano/GPU.
# So just need some extra handling steps after-the-fact.
# It doesn't look like there is any conflict with sending different sentences in, for f_next()

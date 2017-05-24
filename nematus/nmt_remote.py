#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a neural machine translation model with soft attention
"""

import argparse
import os
import time
from collections import OrderedDict

import Pyro4
import numpy
import theano
import theano.tensor as tensor

from nmt_utils import init_params, build_model, build_sampler
import optimizers
from theano_util import load_params, init_theano_params, itemlist, unzip_from_theano, zip_to_theano

profile = False

Pyro4.config.SERIALIZER = 'pickle'


@Pyro4.expose
class RemoteMT(object):
    # TODO: would be nice to use __init__ here... but Pyro does not pass args??
    def init(self, model_options):
        """Exposes: (but Pyro does not see them)
            self.f_init
            self.f_next
            self.f_log_probs
            self.f_grad_shared
            self.f_update
        """

        reload_ = model_options['reload_']
        saveto = model_options['saveto']
        decay_c = model_options['decay_c']
        alpha_c = model_options['alpha_c']
        map_decay_c = model_options['map_decay_c']
        finetune = model_options['finetune']
        finetune_only_last = model_options['finetune_only_last']
        clip_c = model_options['clip_c']
        optimizer = model_options['optimizer']

        comp_start = time.time()

        print 'Building model'
        params = init_params(model_options)
        # reload parameters
        if reload_ and os.path.exists(saveto):
            print 'Reloading model parameters'
            params = load_params(saveto, params)

        self.tparams = init_theano_params(params)

        trng, self.use_noise, x, x_mask, y, y_mask, opt_ret, per_sent_neg_log_prob = build_model(self.tparams, model_options)

        inps = [x, x_mask, y, y_mask]

        self.f_init, self.f_next = build_sampler(self.tparams, model_options, self.use_noise, trng)

        # before any regularizer
        print 'Building f_log_probs...',
        self.f_log_probs = theano.function(inps, per_sent_neg_log_prob, profile=profile)
        print 'Done'

        # apply per-sentence weight to cost_vec before averaging
        per_sent_weight = tensor.vector('per_sent_weight', dtype='float32')
        per_sent_weight.tag.test_value = numpy.ones(10).astype('float32')
        cost = (per_sent_neg_log_prob * per_sent_weight).mean()  # mean of elem-wise multiply

        # apply L2 regularization on weights
        if decay_c > 0.:
            decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
            weight_decay = 0.
            for kk, vv in self.tparams.iteritems():
                weight_decay += (vv ** 2).sum()
            weight_decay *= decay_c
            cost += weight_decay

        # regularize the alpha weights
        if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
            alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
            alpha_reg = alpha_c * (
                (tensor.cast(y_mask.sum(0) // x_mask.sum(0), 'float32')[:, None] -
                 opt_ret['dec_alphas'].sum(0)) ** 2).sum(1).mean()
            cost += alpha_reg

        # apply L2 regularisation to loaded model (map training)
        if map_decay_c > 0:
            map_decay_c = theano.shared(numpy.float32(map_decay_c), name="map_decay_c")
            weight_map_decay = 0.
            for kk, vv in self.tparams.iteritems():
                init_value = theano.shared(vv.get_value(), name=kk + "_init")
                weight_map_decay += ((vv - init_value) ** 2).sum()
            weight_map_decay *= map_decay_c
            cost += weight_map_decay

        # allow finetuning with fixed embeddings
        if finetune:
            updated_params = OrderedDict(
                [(key, value) for (key, value) in self.tparams.iteritems() if not key.startswith('Wemb')])
        elif finetune_only_last:  # allow finetuning of only last layer (becomes a linear model training problem)
            updated_params = OrderedDict(
                [(key, value) for (key, value) in self.tparams.iteritems() if key in ['ff_logit_W', 'ff_logit_b']])
        else:
            updated_params = self.tparams

        print 'Computing gradient...',
        grads = tensor.grad(cost, wrt=itemlist(updated_params))
        print 'Done'

        # apply gradient clipping here
        if clip_c > 0.:
            g2 = 0.
            for g in grads:
                g2 += (g ** 2).sum()
            new_grads = []
            for g in grads:
                new_grads.append(tensor.switch(g2 > (clip_c ** 2),
                                               g / tensor.sqrt(g2) * clip_c,
                                               g))
            grads = new_grads

        # compile the optimizer, the actual computational graph is compiled here
        lr = tensor.scalar(name='lr')

        print 'Building optimizers...',
        op_map = {'adam': optimizers.adam, 'adadelta': optimizers.adadelta,
                  'rmsprop': optimizers.rmsprop, 'sgd': optimizers.sgd}
        inps = inps + [per_sent_weight, ]
        self.f_grad_shared, self.f_update = op_map[optimizer](lr, updated_params, grads, inps, per_sent_neg_log_prob, profile=profile)
        print 'Done'

        print 'Total compilation time: {0:.1f}s'.format(time.time() - comp_start)

    ############ TODO: There must be a better way...

    def x_f_init(self, x, x_mask=None):
        # get initial state of decoder rnn and encoder context
        # state0, ctx0 = f_init(x)
        # x: a [BATCHED?] sequence of word ids followed by 0 (0 = eos id)
        # usage (one sentence at a time): x is size (1, 49, 1), (1, 26, 1), etc.
        # idx 1 is input sentence length, idx 2 is batch
        if x_mask is None:
            x_mask = numpy.ones(x.shape[1:]).astype(numpy.float32)
        return self.f_init(x, x_mask)

    def x_f_next(self, word, ctx, state, x_mask=None):
        # next_prob, next_word, next_state = f_next(word, ctx, state)
        if x_mask is None:
            x_mask = numpy.ones( numpy.shape(ctx)[:-1] ).astype(numpy.float32)
        return self.f_next(word, ctx, state, x_mask)

    def x_f_log_probs(self, x, x_mask, y, y_mask):
        return self.f_log_probs(x, x_mask, y, y_mask)

    def x_f_grad_shared(self, x, x_mask, y, y_mask, per_sent_weight=None, per_sent_cost=False):
        # compute cost, grads and copy grads to shared variables
        # cost = f_grad_shared(x, x_mask, y, y_mask)
        if per_sent_weight is None:
            per_sent_weight = numpy.ones(numpy.array(y).shape[1], dtype=numpy.float32)
        else:
            per_sent_weight = numpy.array(per_sent_weight).astype(numpy.float32)
        cost_vec = self.f_grad_shared(x, x_mask, y, y_mask, per_sent_weight)
        if per_sent_cost:
            return cost_vec
        else:
            return cost_vec.sum()

    def x_f_update(self, lrate):
        # do the update on parameters
        # CALLED AFTER f_grad_shared, which computes gradients
        self.f_update(lrate)

    ############

    def set_noise_val(self, val):
        self.use_noise.set_value(val)

    def get_params_from_theano(self):
        return unzip_from_theano(self.tparams)

    def send_params_to_theano(self, params):
        zip_to_theano(params, self.tparams)

    def print_norms_slow(self):
        params = self.get_params_from_theano()
        for k in params:
            print k, 'norm', numpy.linalg.norm(params[k]), 'max', numpy.max(params[k]), 'min', numpy.min(params[k])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='todo')
    parser.add_argument('--key', type=str, help='hmac key for Pyro')
    parser.add_argument('--name', type=str, help='name to register with Pyro nameserver')
    parser.add_argument('--port', type=int, help='port Pyro nameserver')
    args = parser.parse_args()

    Pyro4.config.NS_PORT = args.port

    daemon = Pyro4.Daemon()                                 # make a Pyro daemon
    # daemon = Pyro4.Daemon(port=args.port)                 # make a Pyro daemon
    daemon._pyroHmacKey = args.key                          # set hmac key of daemon
    ns = Pyro4.locateNS(hmac_key=args.key, port=args.port)  # find the Pyro name server
    uri = daemon.register(RemoteMT)                            # register Pyro object
    ns.register(args.name, uri)                             # register the object with a name in the name server

    print("Ready.")
    daemon.requestLoop()                                    # start the event loop of the server to wait for calls

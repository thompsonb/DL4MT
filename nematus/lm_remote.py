#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a kenlm LM 
"""

import argparse

import Pyro4
import numpy

from lm import lm_factory
from util import seqs2words, deBPE


Pyro4.config.SERIALIZER = 'pickle'


@Pyro4.expose
class RemoteLM(object):
    # TODO: would be nice to use __init__ here... but Pyro does not pass args??
    # load model from specified zip file
    # zip file should contain two files: model (containing model) and model.pkl
    # model.pkl has field model_type specifying model type
    def init(self, model_zip_path, id_to_word):
        self.model = lm_factory(model_zip_path)
        self.id_to_word = id_to_word

    def score(self, x_or_y):
        if len(x_or_y.shape) > 2:  # x shape: (1, N, M). y shape: (N, M)  todo: work with factors
            x_or_y = numpy.squeeze(x_or_y, axis=0)
        """
        Nematus is generally called on 1)Tokenized, 2)Truecased, 3)BPE data.
        So we will train KenLM on Tokenized, Truecase data.
        Therefore all we need to do is convert to a string and deBPE.
        """
        sentences = [deBPE(seqs2words(seq, self.id_to_word)) for seq in x_or_y.T]
        scores = self.model.score(sentences)
        #try:
        #    print 'remote LM sentences/scores:'
        #    for sent, score in zip(sentences, scores):
        #        print '"'+sent+'":', score
        #except Exception, e:
        #    print 'failed to print LM sentences/scores', e
        return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='todo')
    parser.add_argument('--key', type=str, help='hmac key for Pyro')
    parser.add_argument('--name', type=str, help='name to register with Pyro nameserver')
    parser.add_argument('--port', type=int, help='port Pyro nameserver')
    args = parser.parse_args()

    Pyro4.config.NS_PORT = args.port

    daemon = Pyro4.Daemon()                                 # make a Pyro daemon
    daemon._pyroHmacKey = args.key                          # set hmac key of daemon
    ns = Pyro4.locateNS(hmac_key=args.key, port=args.port)  # find the Pyro name server
    uri = daemon.register(RemoteLM)                            # register Pyro object
    ns.register(args.name, uri)                             # register the object with a name in the name server

    print("Ready.")
    daemon.requestLoop()                                    # start the event loop of the server to wait for calls

import Pyro4
import sys
import os

sys.path.insert(1, os.path.abspath('../nematus'))
from nematus.util import load_dict
"""
This initializes the remote process and acts as nmt_client.py's train2 function until line: 187 or
remote.init(model_options)
"""


def initialize(model_options, pyro_port, pyro_name, pyro_key):
    if model_options['dim_per_factor'] is None:
        if model_options['factors'] == 1:
            model_options['dim_per_factor'] = [model_options['dim_word']]
        else:
            sys.stderr.write('Error: if using factored input, you must specify \'dim_per_factor\'\n')
            sys.exit(1)

    assert (len(model_options['dictionaries']) == model_options['factors'] + 1)
    # one dictionary per source factor + 1 for target factor
    assert (len(model_options['dim_per_factor']) == model_options['factors'])
    # each factor embedding has its own dimensionality
    assert (sum(model_options['dim_per_factor']) == model_options[
        'dim_word'])  # dimensionality of factor embeddings sums up to total dimensionality of input embedding vector

    # load dictionaries and invert them
    worddicts = [None] * len(model_options['dictionaries'])
    for ii, dd in enumerate(model_options['dictionaries']):
        worddicts[ii] = load_dict(dd)

    if model_options['n_words_src'] is None:
        n_words_src = len(worddicts[0])
        model_options['n_words_src'] = n_words_src
    if model_options['n_words'] is None:
        n_words = len(worddicts[1])
        model_options['n_words'] = n_words

    print 'Initilizing remote theano server'
    # In order to transfer numpy objects across the network, must use pickle as Pyro Serializer.
    # Also requires various environment flags (PYRO_SERIALIZERS_ACCEPTED, PYRO_SERIALIZER)
    #   for both name server and server.
    Pyro4.config.SERIALIZER = 'pickle'
    Pyro4.config.NS_PORT = pyro_port
    remote = Pyro4.Proxy("PYRONAME:{0}".format(pyro_name))
    remote._pyroHmacKey = pyro_key

    remote.init(model_options)
    return remote

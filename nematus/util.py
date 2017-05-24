"""
Utility functions
"""

import sys
import json
import cPickle as pkl
#import _pickle as pkl # uncomment this line if python3

from copy import deepcopy

def build_model_options(default_model_options, model_dir, lang0, lang1):
    model_options = deepcopy(default_model_options)
    model_options.update(json.loads(open(model_dir+'model.npz.json').read()))
    model_options['saveto'] = model_dir + 'model.npz'
    model_options['dictionaries'] = [model_dir + 'vocab.%s.json' % lang0,
                                     model_dir + 'vocab.%s.json' % lang1]
    # TODO: other paths... datasets, valid_datasets?
    return model_options




#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)


def load_config(basename):
    try:
        with open('%s.json' % basename, 'rb') as f:
            return json.load(f)
    except:
        try:
            with open('%s.pkl' % basename, 'rb') as f:
                return pkl.load(f)
        except:
            sys.stderr.write('Error: config file {0}.json is missing\n'.format(basename))
            sys.exit(1)


def seqs2words(seq, inverse_target_dictionary, warn=True):
    words = []
    for w in seq:
        if w == 0:
            break
        if w in inverse_target_dictionary:
            words.append(inverse_target_dictionary[w])
        else:
            if warn:
                print('WARNING: unknown (sub)word "%s"' % w)
            words.append('UNK')
    return ' '.join(words)


def deBPE(sent):
    return sent.replace('@@ ', '')

#!/usr/bin/env python

import argparse
import os
from lm import KenLM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data = parser.add_argument('kenlm_binary_model')
    data = parser.add_argument('output_file')
    args = parser.parse_args()

    kenlm = KenLM()
    kenlm.wrap_existing_kenlm_model(os.path.abspath(args.kenlm_binary_model))
    kenlm.save(os.path.abspath(args.output_file))



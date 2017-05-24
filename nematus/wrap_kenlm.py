#!/usr/bin/env python

import argparse
import os
from lm import KenLM

if __name__ == '__main__':
    parser = argparse.ArgumentParser('simple script to wrap a kenlm file as expected by rest of code')
    data = parser.add_argument('kenlm_binary_model', help='existing .binlm kenlm file')
    data = parser.add_argument('output_file', help='output file to write, should end in .zip')
    args = parser.parse_args()

    kenlm = KenLM()
    kenlm.wrap_existing_kenlm_model(os.path.abspath(args.kenlm_binary_model))
    kenlm.save(os.path.abspath(args.output_file))



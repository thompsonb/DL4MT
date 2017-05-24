#!/usr/bin/env python

import argparse
import os
from lm import KenLM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text_file', help='input text file to train kenlm on')
    parser.add_argument('output_file', help='output file name (.zip) to write')
    args = parser.parse_args()

    lmx = KenLM()
    lmx.train(os.path.abspath(args.text_file))
    lmx.save(os.path.abspath(args.output_file))

#!/usr/bin/env python

import argparse
import multiprocessing

from dataManager.cache_utils import get_output_name, cached_function_caller

from lm import KenLM


def train_kenlm(input_file):
    output_name = get_output_name([input_file], ['train_kenlm_v0'], extension='zip')

    def worker(output_files):
        output_file = output_files[0]
        lmx = KenLM()
        lmx.train(input_file)
        lmx.save(output_file)

    return cached_function_caller(worker, [output_name])[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data = parser.add_argument('source_file')
    data = parser.add_argument('target_file')
    args = parser.parse_args()

    pool = multiprocessing.Pool(processes=2)
    src_lm, tgt_lm = pool.map(train_kenlm, [args.source_file, args.target_file])
    print 'source lm:', src_lm
    print 'target lm:', tgt_lm

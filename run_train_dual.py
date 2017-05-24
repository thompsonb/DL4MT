#!/usr/bin/env python
import os
import subprocess as sp
import sys

data_dir = sys.argv[1]
lm_a = sys.argv[2]
lm_b = sys.argv[3]

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nematus'))
from nematus.config import python_loc

sp.check_call('%s nematus/train_dual.py %s %s %s'%(python_loc, data_dir, lm_a, lm_b), shell=True)



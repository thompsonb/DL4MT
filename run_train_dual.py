#!/usr/bin/env python
import os
import subprocess as sp
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nematus'))
from nematus.config import python_loc

sp.check_call('%s nematus/train_dual.py'%python_loc, shell=True)



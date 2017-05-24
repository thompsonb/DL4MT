#!/usr/bin/env python
import os
import subprocess as sp
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nematus'))
from nematus.config import python_loc

sp.check_call('%s -m unittest testLL.ParallelSampleTestCase'%python_loc, shell=True)
sp.check_call('%s -m unittest testLL.KenLMTestCase'%python_loc, shell=True)
sp.check_call('%s -m unittest testLL.PyroNematusTests'%python_loc, shell=True)

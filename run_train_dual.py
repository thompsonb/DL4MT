#!/usr/bin/env python
import os
import subprocess as sp
import sys

if len(sys.argv[1]) < 2:
  print "usage run_train_dual.py [CONFIG]"
  sys.exit()

config = sys.argv[1]

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nematus'))
from nematus.config import python_loc

sp.check_call('%s nematus/train_dual.py %s'%(python_loc, config), shell=True)



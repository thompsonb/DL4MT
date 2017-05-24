import os
import sys

sys.path.insert(1, os.path.abspath('../'))
from nematus import shuffle

if __name__ == "__main__":
    shuffle.main(sys.argv[1:])

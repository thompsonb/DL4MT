# Config for a dual-learning run
import os
from nematus.config import WMT16_SYSTEMS_DIR

GPU_ID_MT0 = 0
GPU_ID_MT1 = 1
GPU_ID_LM0 = -1 # place holder
GPU_ID_LM1 = -1 # place holder

DATA_DIR = 'sample_run'
LANG_A = 'en'
LANG_B = 'de'

LM_A = 'testLL/test_data/dummy_lm.zip'  # TODO
LM_B = 'testLL/test_data/dummy_lm.zip'  # TODO

LOGGING_LEVEL = 'WARNING' # CRITICAL, ERROR, WARNING, INFO, DEBUG

MODELDIR_AB = os.path.join(WMT16_SYSTEMS_DIR, '%s-%s/'%(LANG_A, LANG_B))
MODELDIR_BA = os.path.join(WMT16_SYSTEMS_DIR, '%s-%s/'%(LANG_B, LANG_A))

PARALLEL_DATASETS=('testLL/test_data/bi.'+LANG_A, 'testLL/test_data/bi.'+LANG_B)

MONOLINGUAL_DATASETS=('testLL/test_data/mono.'+LANG_A, 'testLL/test_data/mono.'+LANG_B)

VALID_DATASETS=('data/newsdev2016.bpe.ro', 'data/newsdev2016.bpe.en') # TODO!!

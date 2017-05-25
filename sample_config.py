# Config for a dual-learning run
import os
from nematus.config import WMT16_SYSTEMS_DIR

DATA_DIR = 'sample_run'
LANG_A = 'en'
LANG_B = 'de'

LM_A = '/home/bjt/dummy_lm.zip'  # TODO
LM_B = '/home/bjt/dummy_lm.zip'  # TODO

MODELDIR_AB = os.path.join(WMT16_SYSTEMS_DIR, '%s-%s/'%(LANG_A, LANG_B))
MODELDIR_BA = os.path.join(WMT16_SYSTEMS_DIR, '%s-%s/'%(LANG_B, LANG_A))

PARALLEL_DATASETS=('testLL/test_data/bi.'+LANG_A, 'testLL/test_data/bi.'+LANG_B)

MONOLINGUAL_DATASETS=('testLL/test_data/mono.'+LANG_A, 'testLL/test_data/mono.'+LANG_B)

VALID_DATASETS=('data/newsdev2016.bpe.ro', 'data/newsdev2016.bpe.en') # TODO!!

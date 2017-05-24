# Config for a dual-learning run
import os

DATA_DIR = "runs"
LANG_A = 'en'
LANG_B = 'de'
LM_A = "testLL/test_data/en.lm.zip"
LM_B = "testLL/test_data/de.lm.zip"

_WMT16_SYSTEMS_DIR="/export/b09/ws15gkumar/experiments/wmt16-models/" # see also nematus/config.py (used for unit tests)
MODELDIR_AB = os.path.join(_WMT16_SYSTEMS_DIR, '%s-%s/'%(LANG_A, LANG_B))
MODELDIR_BA = os.path.join(_WMT16_SYSTEMS_DIR, '%s-%s/'%(LANG_A, LANG_B))

PARALLEL_DATASETS=(DATA_DIR+'/corpus.'+LANG_A, DATA_DIR+'/corpus.'+LANG_B)
MONOLINGUAL_DATASETS=(DATA_DIR+'/corpus.'+LANG_A, DATA_DIR+'/corpus.'+LANG_B) # HACK - passing in parallel
VALID_DATASETS=('data/newsdev2016.bpe.ro', 'data/newsdev2016.bpe.en')

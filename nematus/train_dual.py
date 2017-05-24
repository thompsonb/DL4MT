from copy import deepcopy
import json
import os
import sys

from nmt_dual_client import train
from util import build_model_options
from nmt_client import default_model_options
from config import wmt16_systems_dir

if __name__ == '__main__':
    LANG_A = 'en'
    LANG_B = 'de'
    DATA_DIR = sys.argv[1]
    LM_A = sys.argv[2]
    LM_B = sys.argv[3]
    modeldir_AB = os.path.join(wmt16_systems_dir, '%s-%s'%(LANG_A, LANG_B))
    modeldir_BA = os.path.join(wmt16_systems_dir, '%s-%s'%(LANG_A, LANG_B))

    model_optionsAB = build_model_options(default_model_options, modeldir_AB, LANG_A, LANG_B)
    model_optionsBA = build_model_options(default_model_options, modeldir_BA, LANG_B, LANG_A)

    validerr = train(model_options_a_b=model_optionsAB,
                     model_options_b_a=model_optionsBA,
                     lrate=0.0001,
                     maxlen=50,
                     # TODO: monolingual data
                     parallel_datasets=(DATA_DIR+'/corpus.'+LANG_A, DATA_DIR+'/corpus.'+LANG_B),
                     monolingual_datasets=(DATA_DIR+'/corpus.'+LANG_A, DATA_DIR+'/corpus.'+LANG_B), # HACK - passing in parallel
                     valid_datasets=('data/newsdev2016.bpe.ro', 'data/newsdev2016.bpe.en'),  # TODO!
                     dictionaries_a_b=[modeldir_AB + '/vocab.%s.json' % LANG_A, modeldir_AB + '/vocab.%s.json' % LANG_B],
                     dictionaries_b_a=[modeldir_BA + '/vocab.%s.json' % LANG_B, modeldir_BA + '/vocab.%s.json' % LANG_A],
                     valid_freq=10000,
                     disp_freq=1000,
                     save_freq=30000,
                     sample_freq=10000,
                     #external_validation_script='./validate.sh',
                     language_models=(LM_A, LM_B),
                     )
    print validerr

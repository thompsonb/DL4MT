from copy import deepcopy
import json
import os
import sys
import imp

from nmt_dual_client import train
from util import build_model_options
from nmt_client import default_model_options

CONFIG_FILE = sys.argv[1]
run_config = imp.load_source('', CONFIG_FILE)

if __name__ == '__main__':
    LANG_A = run_config.LANG_A
    LANG_B = run_config.LANG_B
    DATA_DIR = run_config.DATA_DIR
    LM_A = run_config.LM_A
    LM_B = run_config.LM_B
    modeldir_AB = run_config.MODELDIR_AB
    modeldir_BA = run_config.MODELDIR_BA

    model_optionsAB = build_model_options(default_model_options, modeldir_AB, LANG_A, LANG_B)
    model_optionsBA = build_model_options(default_model_options, modeldir_BA, LANG_B, LANG_A)

    validerr = train(model_options_a_b=model_optionsAB,
                     model_options_b_a=model_optionsBA,
                     lrate=0.0001,
                     maxlen=50,
                     # TODO: monolingual data
                     parallel_datasets=run_config.PARALLEL_DATASETS,
                     monolingual_datasets=run_config.MONOLINGUAL_DATASETS, # HACK - passing in parallel
                     valid_datasets=run_config.VALID_DATASETS,  # TODO!
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

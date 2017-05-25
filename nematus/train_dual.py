from copy import deepcopy
import json
import os
import sys
import imp
import logging

from nmt_dual_client import train
from util import build_model_options
from nmt_client import default_model_options

CONFIG_FILE = sys.argv[1]
run_config = imp.load_source('', CONFIG_FILE)

if __name__ == '__main__':
    
    # get config params
    GPU_ID_MT0 = run_config.GPU_ID_MT0
    GPU_ID_MT1 = run_config.GPU_ID_MT1
    GPU_ID_LM0 = run_config.GPU_ID_LM0
    GPU_ID_LM1 = run_config.GPU_ID_LM1

    LANG_A = run_config.LANG_A
    LANG_B = run_config.LANG_B

    LM_A = run_config.LM_A
    LM_B = run_config.LM_B

    LOGGING_LEVEL = run_config.LOGGING_LEVEL
    logging.basicConfig(level=LOGGING_LEVEL)


    MODELDIR_AB = run_config.MODELDIR_AB
    MODELDIR_BA = run_config.MODELDIR_BA


    model_optionsAB = build_model_options(default_model_options, MODELDIR_AB, LANG_A, LANG_B)
    model_optionsBA = build_model_options(default_model_options, MODELDIR_BA, LANG_B, LANG_A)

    validerr = train(model_options_a_b=model_optionsAB,
                     model_options_b_a=model_optionsBA,
                     lrate=0.0001,
                     maxlen=50,
                     # TODO: monolingual data
                     parallel_datasets=run_config.PARALLEL_DATASETS,
                     monolingual_datasets=run_config.MONOLINGUAL_DATASETS, 
                     valid_datasets=run_config.VALID_DATASETS,  
                     dictionaries_a_b=[MODELDIR_AB + '/vocab.%s.json' % LANG_A, MODELDIR_AB + '/vocab.%s.json' % LANG_B],
                     dictionaries_b_a=[MODELDIR_BA + '/vocab.%s.json' % LANG_B, MODELDIR_BA + '/vocab.%s.json' % LANG_A],
                     valid_freq=10000,
                     disp_freq=1000,
                     save_freq=30000,
                     sample_freq=10000,
                     #external_validation_script='./validate.sh',
                     language_models=(LM_A, LM_B),
                     mt_gpu_ids = (GPU_ID_MT0, GPU_ID_MT1),
                     lm_gpu_ids = (GPU_ID_LM0, GPU_ID_LM1),
                     )
    print validerr

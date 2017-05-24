
This code is based on Nematus (https://github.com/rsennrich/nematus) - please ensure you can run Nematus before proceeding.

# Requirements: #

The default language model is KenLM. You need both kenlm and the python wrapper
(pip install https://github.com/kpu/kenlm/archive/master.zip)

You need a version of python with 
 * Theano (tested with Theano-0.9.0.dev4) 
 * contextlib2 (tested with 0.5.4)
 * Pyro4 (pip install Pyro4) 
 * ipdb (pip install ipdb). 

I used anaconda to install Theano and contextlib2. 

Theano should be compiled for a GPU. Tested with CUDA v ?? and cuDNN 5.1, but probably best to use a newer version.


# Setup: #

There are a number of files that need configuration.

An attempt was made to put all configuration items in nematus/config.py - put paths to your python, CUDA, etc here.

At the moment, configuring the training script requires changing values in nematus/train_dual.py - put paths to your data, initialized Nematus models, and trained language models here.

The language model required here is a kenlm model with a small wrapper around it - use nematus/train_LMs.py to train these models.


# Testing the build: #
./run_tests.py  (THIS WILL FAIL - data paths are all broken)


# Running the code: #
./run_train_dual.py



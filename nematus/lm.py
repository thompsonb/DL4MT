import kenlm
import os
import pickle
import tempfile
import zipfile

import requests

from config import KENLM_PATH
from pyro_utils import BGProc
from config import PORT_NUMBER, TEMP_DIR

import time
import subprocess
import shlex


def _unzip_to_tempdir(model_zip_path):
    # temp folder (automatically deleted on exit)
    tmpdir = tempfile.mkdtemp(dir=TEMP_DIR)
    # unzip model into tempdir
    with zipfile.ZipFile(model_zip_path, 'r', allowZip64=True) as zip_ref:
        zip_ref.extractall(tmpdir)
    return tmpdir


def _zip_to_model(tmpdir, model_zip_path):
    # make pickle file with model options
    # create zipfile archive
    zf = zipfile.ZipFile(model_zip_path, 'w', allowZip64=True)
    zf.compress_type = zipfile.ZIP_DEFLATED  # saw a note that this helps with backwards compat

    # Adding files from directory 'files'
    for _, _, files in os.walk(tmpdir):
        for f in files:
            zf.write(os.path.join(tmpdir, f), f)


class AbstractLM(object):
    def train(self, path_to_text):
        raise NotImplementedError()

    def save(self, model_file_name):
        raise NotImplementedError()

    def load(self, model_file_name):
        raise NotImplementedError()

    def score(self, sentences):
        raise NotImplementedError()

    def _assert_initilized(self):
        if not hasattr(self, 'tmpdir'):
            raise Exception('Did you forget to run train or load first?')


class DummyLM(AbstractLM):
    def train(self, path_to_text):
        pass

    def save(self, model_file_name):
        self.tmpdir = tempfile.mkdtemp(dir=TEMP_DIR)
        params = dict(model_type='dummy')
        pkl_fname = os.path.join(self.tmpdir, 'params.pkl')
        with open(pkl_fname, 'w') as fileObject:
            pickle.dump(params, fileObject)
        _zip_to_model(self.tmpdir, model_file_name)

    def load(self, model_file_name):
        pass

    def score(self, sentences):
        return [-42.0 for _ in sentences]


class KenLM(AbstractLM):
    """
    implements simple wrapper around kenlm
    model is saved as kenlm_model.binary in zip file
    model_type is "kenlm"
    """

    def train(self, path_to_text):
        # also stores binary in temp directory
        self.tmpdir = tempfile.mkdtemp(dir=TEMP_DIR)
        model_arpa_path = os.path.join(self.tmpdir, 'kenlm_model.arpa')
        model_binary_path = os.path.join(self.tmpdir, 'kenlm_model.binary')

        myinput = open(path_to_text)
        myoutput = open(model_arpa_path, 'w') 
        args = shlex.split(os.path.join(KENLM_PATH, 'bin/lmplz') + ' -o 5 -S 40% --skip_symbols </s> <unk>')
        # from kenlm exception: --skip_symbols: to avoid this exception:
        # Special word </s> is not allowed in the corpus.  I plan to support models containing <unk> in the future.
        # Pass --skip_symbols to convert these symbols to whitespace.
        p = subprocess.Popen(args, stdin=myinput, stdout=myoutput)
        p.wait()

        #convert arpa to binary
        p = subprocess.Popen(shlex.split('%s %s %s' % (os.path.join(KENLM_PATH, 'bin/build_binary'), model_arpa_path, model_binary_path)))
        p.wait()

        #remove arpa file
        p=subprocess.Popen(shlex.split('rm %s' % model_arpa_path))
        p.wait()

        #lm_bin = os.path.join(KENLM_PATH, 'bin/lmplz')
        #binarize_bin = os.path.join(KENLM_PATH, 'bin/build_binary')
        #subprocess.check_call('%s -o 5 -S 40%% > %s' % (lm_bin, model_arpa_path))
        #subprocess.check_call('%s %s %s' % (binarize_bin, model_arpa_path, model_binary_path))
        #subprocess.check_call('rm %s' % model_arpa_path)

        self.kenlm_model = kenlm.Model(model_binary_path)

    def save(self, model_file_name):
        """
        save trained model to disk
        TODO (nice to have): write anything that seems useful (training parameters, date trained, etc) to params.pkl
        """
        self._assert_initilized()
        params = dict(model_type='kenlm')
        pkl_fname = os.path.join(self.tmpdir, 'params.pkl')
        with open(pkl_fname, 'w') as fileObject:
            pickle.dump(params, fileObject)

        _zip_to_model(self.tmpdir, model_file_name)

    def load(self, model_file_name):
        self.tmpdir = _unzip_to_tempdir(model_file_name)
        self.kenlm_model = kenlm.Model(os.path.join(self.tmpdir, 'kenlm_model.binary'))

    def score(self, sentences):
        self._assert_initilized()
        return [self.kenlm_model.score(sent, bos=True, eos=True) for sent in sentences]




def lm_factory(model_file_name):
    """
    Peek inside model and see which language model class should open it,
      and return an instantiation of that class, with said model loaded
    :param model_file_name: NematusLL language model file (zip containing params.pkl, etc)
    :return: instantiated language model class (implements AbstractLM interface)
    """
    print 'creating class map'
    class_map = dict(kenlm=KenLM,
                     dummy=DummyLM)

    print 'loading pickle file'
    with zipfile.ZipFile(model_file_name, 'r') as zf:
        with zf.open('params.pkl') as fh:
            params = pickle.load(fh)

    print 'setting model type'
    model_type = params['model_type']
    LM_Class = class_map[model_type]
    lm = LM_Class()

    print 'loading model file'
    lm.load(model_file_name)
    return lm


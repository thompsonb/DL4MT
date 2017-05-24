import os
import random
import signal
import socket
import string
import subprocess as sp
import time
from contextlib import contextmanager

import contextlib2

from config import python_loc, cuda_loc


class BGProc(object):
    """Wrapper to handle background processes: On cleanup, kills the process and all children."""

    def __init__(self, cmd, extra_env_vars=None):
        self.cmd = cmd
        self.extra_env_vars = extra_env_vars or dict()

    def __enter__(self):
        # The os.setpgrp() is passed in the argument preexec_fn so
        # it's run after the fork() and before  exec() to run the shell.
        my_env = os.environ.copy()
        for k, v in self.extra_env_vars.items():
            print('setting %s=%s (expanded to "%s") in environment' % (k, v, os.path.expandvars(str(v))))
            # Allow updates like PATH='/foo/bar/:$PATH'
            my_env[k] = os.path.expandvars(str(v))
        print 'command:', self.cmd
        self.proc = sp.Popen(self.cmd, shell=True, env=my_env, preexec_fn=os.setpgrp)
        time.sleep(5)  # give process a little time to start
        return self.proc

    def __exit__(self, type_, value, traceback):
        print('cleanup: killing group for pid %d' % self.proc.pid)
        os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)


class NameServerSP(BGProc):
    """
    Context manager which sets up environment required by Pyro4 namespace server and runs it in background.
    Performs automatic cleanup on exit.
    """
    def __init__(self, key=None, port=None, host=None):
        cmd = '%s -m Pyro4.naming' % python_loc
        if key is not None:
            cmd += ' --key %s' % key
        if port is not None:
            cmd += ' --port %s' % port
        if host is not None:
            cmd += ' --host %s' % host

        extra_env_vars = dict(PYRO_SERIALIZERS_ACCEPTED='pickle,json',
                              PYRO_SERIALIZER='pickle')
        super(self.__class__, self).__init__(cmd, extra_env_vars)


class remoteSP(BGProc):
    """
    Context manager which sets up Theano environment and runs Pyro4 remote process in background.
    Performs automatic cleanup on exit.
    """
    def __init__(self, remote_script, key=None, name=None, port=None, host=None, gpu_id=0):
        cmd = '%s %s' % (python_loc, remote_script)
        if key is not None:
            cmd += ' --key %s' % key
        if port is not None:
            cmd += ' --port %s' % port
        if host is not None:
            cmd += ' --host %s' % host
        if name is not None:
            cmd += ' --name %s' % name

        # start theano server
        # Note: Theano will let you give it a device (e.g. THEANO_FLAGS=...,device=gpu3) and
        # it will put a small process that device. But it won't actually use it - it will use gpu0.
        # Not sure if this is a Theano "feature" or a Nematus one... but the easy way to handle it seems
        # to be to use CUDA_VISIBLE_DEVICES=<num> to specify the device.
        # Theano/Nemetus will then think (and print) that it is running on gpu0, but it will really be on gpu<num>"""
        extra_env_vars = dict(CUDA_VISIBLE_DEVICES=gpu_id,
                              THEANO_FLAGS='mode=FAST_RUN,floatX=float32,device=gpu,on_unused_input=warn,warn_float64=pdb,optimizer=fast_compile, exception_verbosity=high', #optimizer=None
                              PYRO_SERIALIZERS_ACCEPTED='pickle,json',
                              PYRO_SERIALIZER='pickle',
                              NS_PORT=port,
                              CUDA_HOME=cuda_loc,
                              LD_LIBRARY_PATH='%s/lib64:$LD_LIBRARY_PATH' % cuda_loc,
                              PATH='%s/bin:$PATH' % cuda_loc)

        super(self.__class__, self).__init__(cmd, extra_env_vars)


def get_unused_port():
    """
    Get an empty port for the Pyro nameservr by opening a socket on random port,
    getting port number, and closing it [not atomic, so race condition is possible...]
    Might be better to open with port 0 (random) and then figure out what port it used.
    """
    so = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    so.bind(('localhost', 0))
    _, port = so.getsockname()
    so.close()
    return port


def get_random_key():
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(10))


@contextmanager
def setup_remotes(remote_metadata_list, pyro_port, pyro_key):
    """
    :param remote_metadata_list: list of dictionaries, each containing:
        "script" -  python script to run
        "name" - name used to register with Pyro nameserver
        "gpu_id"  - GPU ID to run on
    """
    with NameServerSP(key=pyro_key, port=pyro_port) as _:
        with contextlib2.ExitStack() as stack:
            for metadata in remote_metadata_list:
                stack.enter_context(remoteSP(metadata['script'],
                                            key=pyro_key,
                                            name=metadata['name'],
                                            port=pyro_port,
                                            gpu_id=metadata['gpu_id']))

            print 'sleeping extra...'
            time.sleep(20) #TODO: why do I need this now?
            print 'assuming everything is ready to use now' # This is a bad assumption. Can we check this?

            yield

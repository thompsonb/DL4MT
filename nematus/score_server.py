#!/usr/bin/python
from http.server import BaseHTTPRequestHandler, HTTPServer
from theanolm.commands import *
from theanolm.exceptions import *
from io import StringIO
from socket import SHUT_RDWR
import argparse
import logging
import sys
import cgi
from traceback import format_tb
import json
from config import PORT_NUMBER


#dummy score function for now; replace with theanolm later
def myscore_old(sent):
    return 0


def myscore(network, args, sent):
        #sent = 'Hello, world.'
        sent_as_file = StringIO(sent)
        output_file = StringIO()
        score._score_text(sent_as_file, network.vocabulary, scorer,
                    output_file, args.log_base, args.subwords, True)
        output = output_file.getvalue()
        output_by_sent = output.split('# Sentence ')[1:]
        return [o.strip().split()[0] for o in output.split('Sentence perplexity:')[1:]]
#        print([float(o.split('Cross entropy (base 10):')[1].strip().split()[0]) for o in output_by_sent])
#        print(output_by_sent[0].split('Cross entropy (base 10):'))
#        return float(output.split('Cross entropy (base 10):')[1].strip().split()[0])


def myscore_sentences(network, args, sentences):
        #sent = 'Hello, world.'
        sent_as_file = StringIO('\n'.join(sentences))
        output_file = StringIO()
        score._score_text(sent_as_file, network.vocabulary, scorer,
                    output_file, args.log_base, args.subwords, True)
        output = output_file.getvalue()
        return float(output.split('Cross entropy (base 10):')[1].strip().split()[0])


#copied from bin/theanolm
# This exception class is not needed, if gpuarray is not used.
try:
    from theano.gpuarray.type import ContextNotDefined
except ImportError:
    class ContextNotDefined(object):
        pass


def exception_handler(exception_type, exception, traceback):
    print("An unexpected {} exception occurred. Traceback will be written to "
          "debug log. The error message was: {}"
          .format(exception_type.__name__, exception))
    logging.debug("Traceback:")
    for item in format_tb(traceback):
        logging.debug(item.strip())
    sys.exit(2)
###end copied portion###


def get_args():
    parser = argparse.ArgumentParser(prog='theanolm')
    subparsers = parser.add_subparsers(
        title='commands',
        help='selects the command to perform ("theanolm command --help" '
             'displays help for the specific command)')

    score_parser = subparsers.add_parser(
        'score', help='score text or n-best lists using a model')
    score.add_arguments(score_parser)
#    score_parser.set_defaults(command_function=score.score)
    score_parser.set_defaults(command_function=score.score_server)
    args = parser.parse_args()
    print(args)
    if hasattr(args, 'command_function'):
        sys.excepthook = exception_handler
        try:
            args.command_function(args)
        except FileNotFoundError as e:
            print('Could not open one of the required files. The error message '
                  'was: ' + str(e))
            sys.exit(2)
        except NumberError as e:
            print('A numerical error has occurred. This may happen e.g. when '
                  'network parameters go to infinity. If this happens during '
                  'training, using a smaller learning rate usually helps. '
                  'Another possibility is to use gradient normalization. If '
                  'this happens during inference, there is probably something '
                  'wrong in the model. The error message was: ' + str(e))
            sys.exit(2)
        except ContextNotDefined as e:
            print('Theano returned error "' + str(e) + '". You need to map the '
                  'device contexts that are used in the architecture file to '
                  'CUDA devices (e.g. '
                  'THEANO_FLAGS=contexts=dev0->cuda0;dev1->cuda1).')
            sys.exit(2)
        except TheanoConfigurationError as e:
            print('There is a problem with Theano configuration. Please check '
                  'THEANO_FLAGS environment variable and .theanorc file. The '
                  'error message was: ' + str(e))
            sys.exit(2)
    return args


#This class will handles any incoming request from
#the browser 
class myHandler(BaseHTTPRequestHandler):

        #Handler for the GET requests
        def do_GET(self):
               f=  open('input_sentence.txt', 'r')
               text = f.read()
               f.close()
               self.send_response(200)
               self.send_header('Content-type','text/html')
               self.end_headers()
                # Send the html message
               self.wfile.write(str.encode('%d'%myscore(text)))
               return

        #based on https://pymotw.com/2/BaseHTTPServer/
        def do_POST(self):
            # Parse the form data posted
            form = cgi.FieldStorage(
                fp=self.rfile, 
                headers=self.headers,
                environ={'REQUEST_METHOD':'POST',
                         'CONTENT_TYPE':self.headers['Content-Type'],
                         })
    
            # Begin the response
            self.send_response(200)
            self.end_headers()
#            self.wfile.write(b'Client: %s\n' % str(self.client_address))
#            self.wfile.write(b'User-agent: %s\n' % str(self.headers['user-agent']))
#            self.wfile.write(b'Path: %s\n' % self.path)
#            self.wfile.write(b'Form data:\n')
#            self.wfile.write(b'Client:')
    
            # Echo back information about what was posted in the form
            for field in form.keys():
                field_item = form[field]
                if field_item.filename:
                        # The field contains an uploaded file
                    file_data = field_item.file.read()
                    file_len = len(file_data)
                    del file_data
                    self.wfile.write('\tUploaded %s as "%s" (%d bytes)\n' % \
                            (field, field_item.filename, file_len))
                else:
                    # Regular form value
#                    self.wfile.write(str.encode('\t%s=%s\n' % (field, form[field].value)))
                    if field == 'sentences':
                        self.wfile.write(str.encode('\n'.join(myscore(network, args, form[field].value))))
            return


if __name__ == '__main__':
        # Create a web server and define the handler to manage the
        # incoming request
        args = get_args()
        (network, scorer) = score.score_server(args)
        HTTPServer.allow_reuse_address = True
        server = HTTPServer(('', PORT_NUMBER), myHandler)
        print('Started httpserver on port ' , PORT_NUMBER)
#        server = HTTPServer(('', args.port_number), myHandler)
#        print('Started httpserver on port ' , args.port_number)

        # Wait forever for incoming htto requests
        server.serve_forever()


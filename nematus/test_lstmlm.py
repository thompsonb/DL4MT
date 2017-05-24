from lm import LSTM_LM
import signal
import time
import sys

#test train function
llm = LSTM_LM()
llm.train('my_dummy_file.txt')
print llm.score(['The cow jumped over the moon'])

#test load function
llm = LSTM_LM()
llm.load('testing_dummyfile')
print llm.score(['The cow jumped over the moon.'])
llm.save('my_lstm_output')

#test save save function
llm = LSTM_LM()
llm.load('my_lstm_output')
print llm.score(['The cow jumped over the moon.'])

#test intenal load function
llm = LSTM_LM()
llm._load_from_h5('') # Place model directory here with language models
print llm.score(['The cow jumped over the moon'])



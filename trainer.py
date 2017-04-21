'''
https://github.com/MattVitelli/GRUV
'''

from __future__ import print_function
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Lambda, Dropout, TimeDistributed, LSTM, Input
from keras import backend as K
from keras import objectives
from keras.optimizers import RMSprop
from keras.utils import generic_utils
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import soundfile
import soundfunc as sf
import vae_model

'''
----------
FUNCTIONS
----------
'''



'''
----------
INPUT VARIABLES
----------
'''
config = sf.get_config()
blocksize = config['blocksize']
compressed_blocksize = config['compressed_blocksize']
seqlen = config['seqlen'] 
win_edge = config['win_edge']
out_step = config['out_step']
sound_file = config['sound_file']
save_file = config['save_file']
batchsize = config['batchsize']
domain = config['domain']

load_model_flag = 1
load_file = sound_file+"_modelsave_4_0"
if load_model_flag == 0: start_epoch = 0
else: start_epoch = 4

sig, samplerate = soundfile.read(sound_file)
print("sample rate: ", samplerate)
print("sig shape: ", sig.shape)
print("flattened sig shape:", sig.shape)

sig_blocks = []
ang_blocks = []

'''
----------
PREPROCESS DATA
----------
'''

print("Dividing into blocks of size :", blocksize)
print("Converting to frequency domain...")
if domain == "rfft":
    sig_blocks, ang_blocks = sf.conv_to_rfft(sig, blocksize, win_edge)
elif domain == "dct":
    sig_blocks = sf.conv_to_dct(sig, blocksize, win_edge, compressed_blocksize)
print("Number of blocks:", len(sig_blocks))
print("Shape of blocks:", np.asarray(sig_blocks[0:len(sig_blocks)-1]).shape)

X_Train = np.zeros((batchsize, seqlen, compressed_blocksize))
y_Train = np.zeros((batchsize, compressed_blocksize))

'''
----------
LOAD / BUILD MODEL
----------
'''
print('Build model...')

model = vae_model.build_model(0.5, batchsize, seqlen, compressed_blocksize) 

print(model.summary())

if load_model_flag==1:
    print('Loading model weights from...', load_file)
    model.load_weights(load_file)

'''
----------
TRAIN THE MODEL
----------
'''

for iteration in range(start_epoch, 6000):
    print()
    print('-' * 50)
    print('Epoch', iteration)
    num_of_its=int((len(sig_blocks)-seqlen-out_step)/batchsize)-2
    print("Number of iterations per epoch:", num_of_its)
    progbar = generic_utils.Progbar(num_of_its)
    losses = []
    for j in range(0,num_of_its):
        X_Train.fill(0)
        y_Train.fill(0)
        for k in range(0,batchsize-1):
            X_Train[k] = sig_blocks[num_of_its*k + j:num_of_its*k + j + seqlen]
            y_Train[k] = sig_blocks[num_of_its*k + j + seqlen + 1]
                    
        loss = model.train_on_batch(X_Train, y_Train)
        losses.append(loss)
        if j==1 or j%50==0:
            progbar.update(j, values=[("loss", np.mean(losses))])
            print()
            print(X_Train.shape)
            print(y_Train.shape)
            losses=[]
        if j%500==0:
            print("saving file...")
            print('iterations:', K.get_value(model.optimizer.iterations))    
            model.save_weights(save_file+str(iteration)+"_"+str(j))
    model.reset_states()

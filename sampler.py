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
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
import numpy as np
import random
import sys
import soundfile
import soundfunc as sf
import vae_model
import argparse

'''
----------
FUNCTIONS
----------
'''

parser = argparse.ArgumentParser(description='sample the music..')
parser.add_argument('-s', default="10_0", metavar='base', type=str,
                    help='suffix of model to be sampled from', required=False, dest='suffix')
parser.add_argument('-o', type=float, default=2, required=False,
                    help='Std Dev on sampling', dest='out_sd')
parser.add_argument('-l', type=int, default=800, required=False,
                    help='length of sequence', dest='outlen')

args = parser.parse_args()

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
batchsize = 1
domain = config['domain']

load_file_suffix= args.suffix
out_sd = args.out_sd
outlen = args.outlen


output_file = load_file_suffix+"_outsd="+str(int(out_sd*100))+"_"+domain+"_"+sound_file
load_file = save_file+load_file_suffix

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
BUILD MODEL
----------
'''

print('Build model...')

model = vae_model.build_model(out_sd, batchsize, seqlen, compressed_blocksize) 

print(model.summary())

'''
----------
LOAD MODEL WEIGHTS
----------
'''
print('Loading model weights from...', load_file)
model.load_weights(load_file)

'''
----------
SEED THE MODEL
----------
'''
rand_start = np.random.randint(0,len(sig_blocks)-15*seqlen)
#"warming up" the stateful model
for feed in range (0,15):
    print(feed)
    seed_sample = sig_blocks[rand_start+seqlen*feed:rand_start+seqlen*(feed+1)]
    #print(np.asarray(seed_sample).shape)
    assert len(seed_sample)==seqlen
    for i in range(0, len(seed_sample)):
        x = np.zeros((batchsize, seqlen, compressed_blocksize))
        if i == seqlen: break
        x[0, i] = seed_sample[i]
    model.predict(x, batch_size=batchsize, verbose=0)

print('Generating sample:')

x = np.zeros((batchsize, seqlen, compressed_blocksize))    

if len(seed_sample)>seqlen: seed_sample=seed_sample[-seqlen:]
generated = []
last_seed = seed_sample

if domain=="rfft":
    for i in range(outlen):  #keep this even
        x = np.zeros((batchsize, seqlen, compressed_blocksize))
        for t in range(0, len(seed_sample)):
            if t == seqlen: break
            x[0, t] = seed_sample[t]

        preds = model.predict_on_batch(x)[0]
        print("preds shape:", np.asarray(preds).shape)
        fft_pred = sf.conv_from_rfft(preds)
        print("fft_pred shape:", np.asarray(fft_pred).shape)
        generated.append(fft_pred)
        print("generated shape:",np.asarray(generated).shape)
        print("seed sample shape:",np.asarray(seed_sample).shape)
        seed_sample=seed_sample[1:]
        seed_sample.append(preds)
        print("----")

    new_gen = np.concatenate(sf.concat_sound_blocks(generated, win_edge))
    print("new-gen shape:", np.asarray(new_gen).shape)

    soundfile.write(output_file, new_gen, samplerate)

elif domain == "dct":
    for i in range(outlen):  #keep this even
        x = np.zeros((batchsize, seqlen, compressed_blocksize))
        for t in range(0, len(seed_sample)):
            if t == seqlen: break
            x[0, t] = seed_sample[t]

        preds = model.predict_on_batch(x)[0]
        print("preds shape:", np.asarray(preds).shape)
        fft_pred = sf.conv_from_dct(preds, len(preds), blocksize)
        print("fft_pred shape:", np.asarray(fft_pred).shape)
        generated.append(fft_pred[0])
        print("generated shape:",np.asarray(generated).shape)
        print("seed sample shape:",np.asarray(seed_sample).shape)
        seed_sample=seed_sample[1:]
        seed_sample.append(preds)
        print("----")

    new_gen = np.concatenate(sf.concat_sound_blocks_mdct(generated, win_edge))
    print("new-gen shape:", np.asarray(new_gen).shape)

    soundfile.write(output_file, new_gen, samplerate)


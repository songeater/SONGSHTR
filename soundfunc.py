import numpy as np
from scipy.fftpack import dct, idct
import sys

'''
----------
FUNCTIONS
----------
'''
def get_config():
    config = {}
    config['sound_file'] = "harvestmoon-mono-hp500.wav"
    config['save_file'] = config['sound_file'] + "_modelsave_"
    config['blocksize']=13000
    config['compressed_blocksize'] = (config['blocksize']//2+1)
    config['seqlen'] = 80 # in number of blocks...
    config['win_edge'] = int(config['blocksize'] / 2)   
    config['out_step'] = 1 # in number of blocks...
    config['batchsize'] = 5 # in number of blocks...
    config['domain'] = "rfft" # either "rfft" or "dct"
    if config['domain'] == "dct": config['win_edge'] = int(config['blocksize'] / 2) # if dct, have to set this
    return config

def concat_sound_blocks_mdct(sound_blocks, edge, clickedge=0):
    print(edge)
    print(np.asarray(sound_blocks).shape)
    new_gen = []
    for i in range(0, len(sound_blocks)-2):
        if i==0:
            new_gen.append(sound_blocks[i][0:-edge-clickedge])
        else:
            temp1 = sound_blocks[i][0:-edge-clickedge]
            temp2 = sound_blocks[i-1][-edge+clickedge:]
            merge = temp1 + temp2
            new_gen.append(merge)
    return new_gen

def conv_to_dct(signal, blocksize, edge, out_blocksize):
    blocks1 = []
    blocks2 = []
    for i in range(0, signal.shape[0]-blocksize-edge, blocksize-edge):
        dct_block = dct(signal[i:i+blocksize], norm='ortho')
        blocks1.append(dct_block)
    if blocksize > out_blocksize:
        for opw in range(len(blocks1)): blocks2.append(blocks1[opw][0:out_blocksize])
    return blocks2

def conv_from_dct(blocks, in_blocksize, out_blocksize):
    new_blocks=[]
    zeropad = [0]*(out_blocksize-in_blocksize)
    dct_pred = blocks
    dct_pred = np.append(dct_pred, zeropad)
    dct_pred = np.asarray(idct(dct_pred, norm='ortho'), dtype=np.float32)
    new_blocks.append(dct_pred)
    return new_blocks

def linear(u):
    return (1-u, u)

def quadratic_out(u):
    u = u * u
    return (1-u, u)

def quadratic_in(u):
    u = 1-u
    u = u * u
    return (u, 1-u)

def linear_bounce(u):
    u = 2 * ( 0.5-u if u > 0.5 else u)
    return (1-u, u)

def merge_sounds(sound1, sound2, fade=linear):
    assert len(sound1)==len(sound2)
    n = len(sound1)
    new_sound = sound1
    for t in range(n):
        u = t / float(n)
        amp1, amp2 = fade(u)
        new_sound[t] = sound1[t]*amp1 + sound2[t]*amp2
    return new_sound

def concat_sound_blocks(sound_blocks, edge):
    print("sound_blocks shape:", np.asarray(sound_blocks[1]).shape)
    new_gen = []
    for i in range(0, len(sound_blocks)-2):
        if i==0: temp1 = sound_blocks[i][0:-edge]
        else: temp1 = sound_blocks[i][edge:-edge]
        new_gen.append(temp1)
        if i%100==0: print("temp1", np.asarray(temp1).shape)
        merge_a = sound_blocks[i] [-edge:]
        merge_b = sound_blocks[i+1][0:edge]
        if edge==0: temp2 = merge_a
        else: temp2 = merge_sounds(merge_a, merge_b)
        if i%100==0: print("temp2", np.asarray(temp2).shape)
        new_gen.append(temp2)
    return new_gen

def conv_to_rfft(signal, blocksize, edge):
    mag_blocks = []
    ang_blocks = []
    for i in range(0, signal.shape[0]-blocksize-edge, blocksize-edge):
        fft_block = np.fft.rfft(signal[i:i+blocksize], norm='ortho')
        mag_blocks.append(np.abs(fft_block))
        ang_blocks.append(np.angle(fft_block))
    return mag_blocks, ang_blocks

def conv_from_rfft(mag_blocks, ang_blocks=0):
    new_blocks=[]
    if ang_blocks==0:
        fft_pred = []
        for opq in range(len(mag_blocks)):
            fft_x = np.cos(0)*mag_blocks[opq]
            fft_y = np.sin(0)*mag_blocks[opq]
            fft_pred.append(fft_x + 1.0j*fft_y)
        new_blocks = np.asarray(np.fft.irfft(mag_blocks, norm='ortho'), dtype=np.float32)
        print("new_blocks shape:", new_blocks.shape)
    else:
        for opq in range(len(mag_blocks)):
            fft_x = np.cos(ang_blocks[opq])*mag_blocks[opq]
            fft_y = np.sin(ang_blocks[opq])*mag_blocks[opq]
            fft_pred = fft_x + 1.0j*fft_y
            fft_pred = np.asarray(np.fft.irfft(fft_pred, norm='ortho'), dtype=np.float32)
            new_blocks.append(fft_pred)
    return new_blocks

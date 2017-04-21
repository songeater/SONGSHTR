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

def build_model(epsilon_std, batchsize, seqlen, blocksize):
    dropout_val = 0.1
    lstm_dims = 128
    latent_dims = lstm_dims

    lstm_input = Input(batch_shape=((batchsize, seqlen, blocksize)))

    lstm_model = TimeDistributed(Dense(lstm_dims))(lstm_input)
    lstm_model = LSTM(lstm_dims, stateful=True, return_sequences=False)(lstm_model)
    lstm_model = Dropout(dropout_val)(lstm_model)
    #lstm_model = LSTM(lstm_dims, stateful=True, return_sequences=True)(lstm_model)
    #lstm_model = Dropout(dropout_val)(lstm_model)
    #lstm_model = LSTM(lstm_dims, stateful=True, return_sequences=False)(lstm_model)
    #lstm_model = Dropout(dropout_val)(lstm_model)

    z_mean = Dense(latent_dims, activation = 'relu')(lstm_model)
    z_log_var = Dense(latent_dims, activation = 'sigmoid')(lstm_model)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batchsize, latent_dims), mean=0.,std=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    vae = Lambda(sampling)([z_mean, z_log_var])
    vae = Dense(lstm_dims, activation = 'relu')(vae)
    vae = Dense(blocksize, activation = 'sigmoid')(vae)

    model = Model(input=lstm_input, output=vae)
    optimizer = RMSprop(lr=0.002, decay = 0.0005, clipvalue=5)

    def vae_loss(x, x_decoded_mean):
        mean_loss = objectives.mean_squared_logarithmic_error(x, x_decoded_mean)
        kl_loss = - 0.25 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return mean_loss + kl_loss

    model.compile(loss=vae_loss, optimizer=optimizer)

    return model

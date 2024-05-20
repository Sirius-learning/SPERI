
from  __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dropout, Activation, Dense, Input, Masking, BatchNormalization, Reshape, Add
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.regularizers import l2
from GRUD_model import GRUD 
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from dataProcess import split_time_series_data
from tensorflow.keras.layers import RepeatVector, Lambda, Permute

def init_gru_state(batch_size, num_hiddens):
    return tf.zeros((batch_size, num_hiddens))

from tensorflow.keras.layers import RepeatVector, Concatenate

def create_grud_model_with_encoder_decoder(output_activation,
                                           use_bidirectional_rnn=False,
                                           use_batchnorm=False,
                                           num_hiddens=64,
                                           input_size=96,
                                           latent_dim=64):

    # Input
    input_x = Input(shape=(input_size, 1), name='x')
    input_m = Input(shape=(input_size, 1), name='m')
    input_d = Input(shape=(input_size, 1), name='d')
    input_xpie = Input(shape=(input_size, 1), name='xpie')
    input_s = Input(shape=(input_size, 1), name='seasonal')
    input_t = Input(shape=(input_size, 1), name='trend')
    input_irradiance = Input(shape=(input_size, 1), name='irradiance')

    input_list = [input_x, input_m, input_d, input_xpie, input_s, input_t]
    encoder_input_list = [input_x, input_m, input_d, input_xpie, input_s, input_t, input_irradiance]

    # Encoder
    grud_layer = GRUD(dropout=0.3, dropout_type='mloss', hidden_size=num_hiddens)
    ht = init_gru_state(batch_size=input_size, num_hiddens=num_hiddens)
    grud_output = grud_layer(input_list, state=ht)
    concatenated_input = Concatenate(axis=-1)([grud_output, input_irradiance])
    mlp_output = Dense(units=latent_dim*1, kernel_regularizer=l2(1e-4))(concatenated_input)
    encoder_output = Reshape((1, latent_dim))(mlp_output)  # 将输出调整为 (batch_size, 1, latent_dim)


    # Decoder
    irradiance_dense = Dense(units=latent_dim, activation='relu')(input_irradiance)
    irradiance_expanded = Reshape((1, latent_dim))(irradiance_dense)
    add_tensors = Add()([encoder_output, irradiance_expanded])
    decoder_output = Dense(units=input_size, activation='relu')(add_tensors)


    # model = Model(inputs=[input_x, input_m, input_d, input_xpie, input_s, input_t, input_irradiance,
    #                       decoder_input_irradiance, decoder_input_state],
    #               outputs=decoder_output, name='grud_model_with_encoder_decoder')

    # return model

# Example usage:
input_size = 96
latent_dim = 64
grud_encoder_decoder_model = create_grud_model_with_encoder_decoder(output_activation='sigmoid',
                                                                     num_hiddens=64,
                                                                     input_size=input_size,
                                                                     latent_dim=latent_dim)
grud_encoder_decoder_model.summary()



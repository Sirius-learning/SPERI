from  __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import math
import time
from tensorflow.keras.layers import Dropout, Activation, Dense, Input, Masking
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.regularizers import l2
#from keras.utils.generic_utils import custom_object_scope




class GRUD(tf.keras.Model):
    def __init__(self, input_size=96,
                 hidden_size=96, 
                 output_size=96, x_mean=0,\
                 bias=True, batch_first=False, bidirectional=False, dropout_type='mloss', dropout=0):
        super(GRUD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_mean = tf.Variable(x_mean, trainable=False, dtype=tf.float32)
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        # num_directions = 2 if bidirectional else 1

        
        self._all_weights = []
        
        # decay rates gamma
        self.w_dg_x = self.add_weight(shape=(input_size,1), initializer='zeros', name='w_dg_x',trainable=True)
        self.w_dg_h = self.add_weight(shape=(1,hidden_size), initializer='zeros', name='w_dg_h',trainable=True)

        # z
        self.w_xz = self.add_weight(shape=(1,hidden_size), initializer='zeros', name='w_xz',trainable=True)
        self.w_hz = self.add_weight(shape=(hidden_size,hidden_size), initializer='zeros', name='w_hz',trainable=True)
        self.w_mz = self.add_weight(shape=(1,hidden_size), initializer='zeros', name='w_mz',trainable=True) #wait

        # r
        self.w_xr = self.add_weight(shape=(1,hidden_size), initializer='zeros', name='w_xr',trainable=True)
        self.w_hr = self.add_weight(shape=(hidden_size,hidden_size), initializer='zeros', name='w_hr',trainable=True)
        self.w_mr = self.add_weight(shape=(1,hidden_size), initializer='zeros', name='w_mr',trainable=True)#wait

        # h_tilde
        self.w_xh = self.add_weight(shape=(1,hidden_size), initializer='zeros', name='w_xh',trainable=True)
        self.w_hh = self.add_weight(shape=(hidden_size,hidden_size), initializer='zeros', name='w_hh',trainable=True)
        self.w_mh = self.add_weight(shape=(1,hidden_size), initializer='zeros', name='w_mh',trainable=True)#wait

        # y (output)
        self.w_hy = self.add_weight(shape=(hidden_size, hidden_size), initializer='zeros', name='w_hy',trainable=True)#wait

        # define bias
        self.b_dg_x = self.add_weight(shape=(input_size,1), initializer='zeros', name='b_dg_x',trainable=True)
        self.b_dg_h = self.add_weight(shape=(input_size,hidden_size), initializer='zeros', name='b_dg_h',trainable=True)
        self.b_z = self.add_weight(shape=(input_size,hidden_size), initializer='zeros', name='b_z',trainable=True)
        self.b_r = self.add_weight(shape=(input_size,hidden_size), initializer='zeros', name='b_r',trainable=True)
        self.b_h = self.add_weight(shape=(input_size,hidden_size), initializer='zeros', name='b_h',trainable=True)
        self.b_y = self.add_weight(shape=(input_size,hidden_size), initializer='zeros', name='b_y',trainable=True)#wait

        #define seasonal, trend weight
        self.w_s = self.add_weight(shape=(1,hidden_size), initializer='zeros', name='w_s',trainable=True)
        self.w_t = self.add_weight(shape=(1,hidden_size), initializer='zeros', name='w_t',trainable=True)
        self.w_r = self.add_weight(shape=(hidden_size,hidden_size), initializer='zeros', name='w_r',trainable=True)


        def get_layer_params(self):
            layer_params = self.weights
            param_names = [var.name for var in layer_params]

            return layer_params, param_names   

        layer_params, param_names = get_layer_params(self)

        for name, param in zip(param_names, layer_params):
            setattr(self, name, param)
        self._all_weights.append(param_names)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.trainable_variables:
            # 使用均匀分布进行初始化
            initializer = tf.keras.initializers.RandomUniform(minval=-stdv, maxval=stdv)
            weight.assign(initializer(weight.shape))
    
    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
    
        def check_hidden_size(self, hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.shape) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.shape)))
        
        check_hidden_size(hidden, expected_hidden_size)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)


    # def get_config(self):
    #     config = super(GRUD, self).get_config()
    #     config.update({
    #         'input_size': self.input_size,
    #         'hidden_size': self.hidden_size,
    #         'output_size': self.output_size,
    #         'x_mean': float(self.x_mean.numpy()),  # Convert to Python float
    #         'bias': self.bias,
    #         'batch_first': self.batch_first,
    #         'bidirectional': self.bidirectional,
    #         'dropout_type': self.dropout_type,
    #         'dropout': self.dropout,
    #         'all_weights':self._all_weights
    #     })

        return config
    
    @property
    def _flat_weights(self):
        return list(self.variables)
    
    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]
    
    
    def call(self, data, state, training=None):

        #h_t = tf.zeros([self.hidden_size], dtype=tf.float32)

        h_t = state
        #Hidden_State = tf.Variable(tf.zeros([self.input_size], dtype=tf.float32)) #input_size=0 in here
        #print("value of ht:",h_t)
        #step_size = data.shape[1]
        #step_size = np.size(data,1)
        step_size=tf.shape(data)[1]
        output = None
        #h_t = Hidden_State
        outputs = []
        states = tf.TensorArray(tf.float32, size=step_size)
        #get variable from model
        w_dg_x = getattr(self, 'w_dg_x')
        w_dg_h = getattr(self, 'w_dg_h')
        w_xz = getattr(self, 'w_xz')
        w_hz = getattr(self, 'w_hz')
        w_mz = getattr(self, 'w_mz')
        w_xr = getattr(self, 'w_xr')
        w_hr = getattr(self, 'w_hr')
        w_mr = getattr(self, 'w_mr')
        w_xh = getattr(self, 'w_xh')
        w_hh = getattr(self, 'w_hh')
        w_mh = getattr(self, 'w_mh')
        w_hy = getattr(self, 'w_hy')

        b_dg_x = getattr(self, 'b_dg_x')
        b_dg_h = getattr(self, 'b_dg_h')
        b_z = getattr(self, 'b_z')
        b_r = getattr(self, 'b_r')
        b_h = getattr(self, 'b_h')
        b_y = getattr(self, 'b_y')
        w_s = getattr(self, 'w_s')
        w_t = getattr(self, 'w_t')
        w_r = getattr(self, 'w_r')
        print("dataSize:",tf.shape(data))
        #print("show data:",data) #here x is right
        for i in range(step_size):
            x = data[0][i]
            m = data[1][i]
            d = data[2][i]
            xpie = data[3][i]
            seasonal = data[4][i]
            trend = data[5][i]
            #print("xpiesize:",tf.shape(xpie))
            #Input decay term
            #print("showm:",tf.transpose(m))
            #print("showx:",tf.transpose(x))
            gamma_x = tf.exp(-tf.maximum(0.0, (w_dg_x * d + b_dg_x))) #(?,96)
            #Hidden state decay term
            gamma_h = tf.exp(-tf.maximum(0.0, (tf.matmul(d , w_dg_h)  + b_dg_h))) #(96,96)
            #print("gammaH:",gamma_x)
            #print("value of m:", tf.transpose(xpie))
            #x = m * x + (1 - m) * (gamma_x * xpie + (1 - gamma_x) * self.x_mean)
            x = m * x + (1 - m) * (gamma_x * xpie)

            print("value of x:",tf.transpose(x))
            #print("value of m:",tf.transpose(x))
            if self.dropout == 0:
                h_t = gamma_h * h_t  #ht:(96,64)
                # reset gate
                r_t = tf.sigmoid(tf.matmul(w_xr , x) + tf.matmul(w_hr , h_t) + tf.matmul(w_mr , m) + b_r)
                # 计算更新门
                z_t = tf.sigmoid(tf.matmul(w_xz , x) + tf.matmul(w_hz , h_t) + tf.matmul(w_mz , m) + b_z)
                # 计算候选隐藏状态
                h_tilde = tf.tanh(tf.matmul(w_xh , x) + tf.matmul(w_hh , (r_t * h_t)) + tf.matmul(w_mh , m) + b_h)
                # 计算隐藏状态
                h_t = (1 - z_t) * h_t + z_t * h_tilde

            elif self.dropout_type == 'Moon':
                '''
                RNNDROP: a novel dropout for rnn in asr(2015)
                '''
                h_t = gamma_h * h_t
                # reset gate
                r_t = tf.sigmoid(tf.matmul(w_xr , x) + tf.matmul(w_hr , h_t) + tf.matmul(w_mr , m) + b_r)
                # 计算更新门
                z_t = tf.sigmoid(tf.matmul(w_xz , x) + tf.matmul(w_hz , h_t) + tf.matmul(w_mz , m) + b_z)
                # 计算候选隐藏状态
                h_tilde = tf.tanh(tf.matmul(w_xh , x) + tf.matmul(w_hh , (r_t * h_t)) + tf.matmul(w_mh , m) + b_h)
                # 计算隐藏状态
                h_t = (1 - z_t) * h_t + z_t * h_tilde
                dropout_layer = tf.keras.layers.Dropout(rate=self.dropout)
                h_t = dropout_layer(h_t)
            elif self.dropout_type == 'Gal':
                '''
                A Theoretically grounded application of dropout in recurrent neural networks(2015)
                '''
                dropout_layer = tf.keras.layers.Dropout(rate=self.dropout)
                h_t = dropout_layer(h_t , training=True)
                h_t = gamma_h * h_t
                h_t = tf.squeeze(h_t)
                # reset gate
                r_t = tf.sigmoid(tf.matmul(w_xr , x) + tf.matmul(w_hr , h_t) + tf.matmul(w_mr , m) + b_r)
                # 计算更新门
                z_t = tf.sigmoid(tf.matmul(w_xz , x) + tf.matmul(w_hz , h_t) + tf.matmul(w_mz , m) + b_z)
                # 计算候选隐藏状态
                h_tilde = tf.tanh(tf.matmul(w_xh , x) + tf.matmul(w_hh , (r_t * h_t)) + tf.matmul(w_mh , m) + b_h)
                # 计算隐藏状态
                h_t = (1 - z_t) * h_t + z_t * h_tilde
            elif self.dropout_type == 'mloss':
                '''
                recurrent dropout without memory loss arXiv 1603.05118
                g = h_tilde, p = the probability to not drop a neuron
                '''
                #h_t = gamma_h * h_t
                h_t = tf.multiply(gamma_h , h_t)
                #print("show gammaH:",gamma_h)
                #print("show ht:",h_t)
                #print("htshape:",h_t.shape)
                # if h_t.shape[0]==1:
                #     h_t = tf.squeeze(h_t , axis=0)
                # reset gate
                r_t = tf.sigmoid(tf.matmul(x , w_xr) + tf.matmul(h_t , w_hr) + tf.matmul(m , w_mr) + b_r)
                # 计算更新门
                z_t = tf.sigmoid(tf.matmul(x , w_xz) + tf.matmul(h_t , w_hz) + tf.matmul(m , w_mz) + b_z)
                # 计算候选隐藏状态
                h_tilde = tf.tanh(tf.matmul(x , w_xh) + tf.matmul((r_t * h_t) , w_hh) + tf.matmul(m , w_mh) + b_h)
                dropout_layer = tf.keras.layers.Dropout(rate=self.dropout)
                #计算隐藏状态
                h_tilde = dropout_layer(h_tilde)
                h_t = (1 - z_t) * h_t + z_t * h_tilde
                #print("show ht:",h_t)
            else:
                h_t = gamma_h * h_t
                # reset gate
                r_t = tf.sigmoid(tf.matmul(w_xr , x) + tf.matmul(w_hr , h_t) + tf.matmul(w_mr , m) + b_r)
                # 计算更新门
                z_t = tf.sigmoid(tf.matmul(w_xz , x) + tf.matmul(w_hz , h_t) + tf.matmul(w_mz , m) + b_z)
                # 计算候选隐藏状态
                h_tilde = tf.tanh(tf.matmul(w_xh , x) + tf.matmul(w_hh , (r_t * h_t)) + tf.matmul(w_mh , m) + b_h)
                # 计算隐藏状态
                h_t = (1 - z_t) * h_t + z_t * h_tilde
            # print("h_t:",h_t.shape)
            # print("hidden size:",self.hidden_size)
            # print("seasonal:",seasonal.shape)
            # print("trend:",trend.shape)
            grud_output = tf.matmul(h_t , w_hy) + b_y
            #grud_output = h_t
            grud_output = tf.sigmoid(grud_output)
            current_output = tf.matmul(grud_output , w_r) + tf.matmul(seasonal , w_s) + tf.matmul(trend , w_t)
            states = states.write(i, current_output)
        # 计算输出
        # tf.add_to_collection('tensor',outputs)
        #print("value of w_hy,b_y:",w_hy,b_y)
        #print("value of h_t:",h_t)
        # tensors_collection = tf.get_collection('tensor')
        # outputs = tensors_collection[0]
        #output = tf.concat(final_output, axis=1)
        return states.stack()
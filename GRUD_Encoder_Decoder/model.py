from  __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dropout, Activation, Dense, Input, Masking, BatchNormalization, Reshape, Add, Concatenate, GlobalAveragePooling1D, Flatten
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.regularizers import l2
#from keras.utils.generic_utils import custom_object_scope
from GRUD_model import GRUD 
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from dataProcess import split_time_series_data


_all_ =  ['create_grud_model', 'load_grud_model']

def create_grud_model(output_activation,
                      predefined_model=None,
                      use_bidirectional_rnn=False, use_batchnorm=False, num_hiddens=64,input_size=96,latent_dim=64
                      ):

    # Input
    input_x = Input(shape=(input_size, 1), name='x')
    input_m = Input(shape=(input_size, 1), name='m')
    input_d = Input(shape=(input_size, 1), name='d')
    input_xpie = Input(shape=(input_size, 1), name='xpie')
    input_irradiance = Input(shape=(input_size, 1), name='irradiance')
    
    input_list = [input_x, input_m, input_d, input_xpie]
    encoder_input_list = [input_x, input_m, input_d, input_xpie, input_irradiance]

    # Encoder
    grud_layer = GRUD(dropout=0.3, dropout_type='mloss', hidden_size=num_hiddens)
    ht = init_gru_state(batch_size=input_size, num_hiddens=num_hiddens)
    grud_output = grud_layer(input_list, state=ht)
    grud_output = Dense(units= 16, activation='relu', name="grud1_16" )(grud_output)
    grud_output = Dense(units=32, activation='relu', name="grud16_32")(grud_output)
    process_irradiance = Dense(units= 16, activation='relu', name="irradiance1_16" )(input_irradiance)
    process_irradiance = Dense(units=32, activation='relu', name="irradiance16_32")(process_irradiance)
    concatenated_input = Add()([grud_output, process_irradiance])
    concatenated_input = Dense(units=32, activation='relu', name="concatenated32")(concatenated_input)
    concatenated_input = Dense(units=64, activation='relu', name="concatenated64")(concatenated_input)
    polling_output = GlobalAveragePooling1D()(concatenated_input)
    encoder_output = polling_output
    #encoder_output = tf.expand_dims(polling_output, 1)
    #encoder_output = Reshape(( 1, latent_dim))(mlp_output)  # 将输出调整为 (batch_size, 1, latent_dim)

    # Decoder
    irradiance_dense = Dense(units=16, activation='relu')(input_irradiance)
    irradiance_dense = Dense(units=latent_dim, activation='relu',name='irradiance_dense')(irradiance_dense)
    #irradiance_expanded = tf.expand_dims(irradiance_dense, 1)
    #irradiance_expanded = Reshape((1, latent_dim))(irradiance_dense)
    add_tensors = Add()([encoder_output, irradiance_dense])
    decoder_output = Dense(units=32, activation='relu')(add_tensors)
    decoder_output = Dense(units=1, activation='relu', name='32one')(decoder_output)
    decoder_output = Flatten()(decoder_output)
    output_list = [decoder_output]
    model = Model(inputs=encoder_input_list, outputs=output_list)
    return model

# def _get_scope_dict():
#     from . import activations, callbacks, grud_layers, layers

#     merge_dict = lambda x, y: dict(list(x.items()) + list(y.items()))
#     scope_dict = {}
#     scope_dict = merge_dict(scope_dict, activations._get_activations_scope_dict())
#     scope_dict = merge_dict(scope_dict, callbacks._get_callbacks_scope_dict())
#     scope_dict = merge_dict(scope_dict, grud_layers._get_grud_layers_scope_dict())
#     scope_dict = merge_dict(scope_dict, layers._get_layers_scope_dict())
#     return scope_dict

# def load_grud_model(file_name):
#     with custom_object_scope(_get_scope_dict()):
#         model = load_model(file_name)
#     return model


def init_gru_state(batch_size, num_hiddens):
    return tf.zeros((batch_size, num_hiddens))



def fit(model, criterion, learning_rate, train_data, dev_data, test_data, ylabel, learning_rate_decay=0, n_epochs=30):
    epoch_losses = []

    # TensorFlow的优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_label, dev_label, test_label = split_time_series_data(ylabel)
    for epoch in range(n_epochs):
        if learning_rate_decay != 0 and epoch % learning_rate_decay == 0:
            learning_rate = learning_rate / 2
            optimizer.learning_rate.assign(learning_rate)
            print('at epoch {} learning_rate is updated to {}'.format(epoch, learning_rate))

        # 训练模型
        losses, acc = [], []
        label, pred = [], []
        y_pred_col = []

        # 前向传播
        with tf.GradientTape() as tape:
            y_pred = model(train_data, training=True)
            y_pred = tf.squeeze(y_pred)
            loss = criterion(train_label, y_pred)

            # 计算准确率
            acc.append(tf.math.equal(tf.cast(tf.math.greater(tf.sigmoid(y_pred), 0.5), tf.float32), train_label))

            # 计算损失
            losses.append(loss.numpy())

            # 保存预测和标签
            y_pred_col.append(y_pred.numpy())
            pred.append(tf.math.greater(y_pred, 0.5))
            label.append(train_label)

            # 反向传播和优化
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_acc = np.mean(np.concatenate(acc))
        train_loss = np.mean(losses)

        # 验证损失
        losses, acc = [], []
        label, pred = [], []
        
        y_pred = model(dev_data, training=False)
        y_pred = tf.squeeze(y_pred)
        loss = criterion(dev_label, y_pred)

        acc.append(tf.math.equal(tf.cast(tf.math.greater(tf.sigmoid(y_pred), 0.5), tf.float32), dev_label))
        losses.append(loss.numpy())

        pred.append(y_pred.numpy())
        label.append(dev_label)

        dev_acc = np.mean(np.concatenate(acc))
        dev_loss = np.mean(losses)

        # 测试损失
        losses, acc = [], []
        label, pred = [], []

        y_pred = model(test_data, training=False)
        y_pred = tf.squeeze(y_pred)
        loss = criterion(test_label, y_pred)

        acc.append(tf.math.equal(tf.cast(tf.math.greater(tf.sigmoid(y_pred), 0.5), tf.float32), test_label))
        losses.append(loss.numpy())

        pred.append(y_pred.numpy())
        label.append(test_label)

        test_acc = np.mean(np.concatenate(acc))
        test_loss = np.mean(losses)

        epoch_losses.append([
            train_loss, dev_loss, test_loss,
            train_acc, dev_acc, test_acc,
            np.concatenate(pred), np.concatenate(pred), np.concatenate(pred),
            np.concatenate(label), np.concatenate(label), np.concatenate(label),
        ])

        pred = np.concatenate(pred)
        label = np.concatenate(label)

        auc_score = tf.keras.metrics.AUC()(label, pred).numpy()

        print("Epoch: {} Train loss: {:.4f}, Dev loss: {:.4f}, Test loss: {:.4f}, Test AUC: {:.4f}".format(
            epoch, train_loss, dev_loss, test_loss, auc_score))

    return epoch_losses


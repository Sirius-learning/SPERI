import tensorflow as tf
import numpy as np
from datetime import datetime
import os
import csv

import pandas as pd
from dataProcess import create_time_interval, create_mask, count_parameters, split_time_series_data, find_previous_observation, r_squared
from model import create_grud_model, fit
from data_handler import DataHandler
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

print("GPU Situation:")
tf.config.experimental.list_physical_devices('GPU')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.compat.v1.Session(config = config)

print("++++++++++++++++++++++++++++++++++++++++++")
ini_miss_rate = 0.2
for i in range(0,7):
    miss_rate = ini_miss_rate + 0.1 * i 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    print('Timestamp: {}'.format(timestamp))
    current_path = os.getcwd()
    result_record_path = os.path.join(current_path,"results_score.txt")
    data_path=os.path.join(current_path,"data","SDU_GRU_pre.csv")
    result_path=os.path.join(current_path,"data","GRUD_Result.csv")
    temp_path=os.path.join(current_path,"data","check.csv")
    result_data_path=os.path.join(current_path,"data","resultdata.csv")
    # seasonal_data_path = os.path.join(current_path, "data", "SDU_PVdata_consective_seasonal_" + str(int(miss_rate * 100)) + "%.csv")
    # trend_data_path = os.path.join(current_path, "data", "SDU_PVdata_consective_trend_" + str(int(miss_rate * 100)) + "%.csv")
    # residual_data_path = os.path.join(current_path, "data", "SDU_PVdata_consective_residual_" + str(int(miss_rate * 100)) + "%.csv")
    
    seasonal_data_path = os.path.join(current_path, "data", "seasonal_data_" + str(int(miss_rate * 100)) + "%.csv")
    trend_data_path = os.path.join(current_path, "data", "trend_data_" + str(int(miss_rate * 100)) + "%.csv")
    residual_data_path = os.path.join(current_path, "data", "SDU_GRU_pre_" + str(int(miss_rate * 100)) + "%.csv")
    
    predict_data_path = os.path.join(current_path,"data","predict_data.csv")
    label_data_path=os.path.join(current_path,"data","SDU_PVdata_consective_label_data.csv")
    normalized_data_path = os.path.join(current_path,"data","SDU_PVdata_normalized.csv")
    data_residual = pd.read_csv(residual_data_path,  header=None)
    seasonal_data = pd.read_csv(seasonal_data_path,header=None)
    trend_data = pd.read_csv(trend_data_path,header=None)
    label_data = pd.read_csv(label_data_path, header=None)
    normalized_data = pd.read_csv(normalized_data_path)
    normalized_data = normalized_data['realPower'].values.reshape(731,96)
    #another PVdata
    pvdata_seasonal_path = os.path.join(current_path, "data", "pvdata_seasonal_" + str(int(miss_rate * 100)) + "%.csv")
    pvdata_trend_path = os.path.join(current_path, "data", "pvdata_trend_" + str(int(miss_rate * 100)) + "%.csv")
    pvdata_residual_path = os.path.join(current_path, "data", "pvdata_residual_" + str(int(miss_rate * 100)) + "%.csv")
    pvdata_label_path=os.path.join(current_path,"data","pvdata_label_data.csv")
    pvdata_normalized_path = os.path.join(current_path,"data","pvdata_normalized.csv")
    pvdata_normalized = pd.read_csv(pvdata_normalized_path)
    pvdata_normalized = pvdata_normalized[:-1]
    pvdata_normalized = pvdata_normalized['realPower'].values.reshape(365,96)
    pvdata_label = pd.read_csv(pvdata_label_path, header=None)
    pvdata_residual = pd.read_csv(pvdata_residual_path, header=None)
    pvdata_seasonal = pd.read_csv(pvdata_seasonal_path,header=None)
    pvdata_trend = pd.read_csv(pvdata_trend_path,header=None)
    pv_x =np.array(pvdata_residual)
    pv_s = np.array(pvdata_seasonal)
    pv_t = np.array(pvdata_trend)
    pv_label = np.array((pvdata_label))
    pv_m = create_mask(pv_x)
    pv_d = create_time_interval(pv_m)
    pv_xpie = find_previous_observation(pv_x,pv_m)
    pv_xzero = np.where(np.isnan(pv_x),0,pv_x)
    pvdata = [pv_xzero, pv_m, pv_d, pv_xpie, pv_s, pv_t, pv_label]

    #numpy transform
    row_x = np.array(data_residual)
    row_s = np.array(seasonal_data)
    row_t = np.array(trend_data)
    label = np.array(label_data)
    true_data = np.array(normalized_data)
    #print("row s:",row_s.shape)
    print("y size:",label.shape)
    # #data normalize
    # x, min_val_x, max_val_x = normalize_data(row_x)
    # s, min_val_s, max_val_s = normalize_data(row_s)
    # t, min_val_t, max_val_t = normalize_data(row_t)
    # label, min_val_y, max_val_y = normalize_data(label)

    #x data process
    m = create_mask(row_x)
    d = create_time_interval(m)
    xpie = find_previous_observation(row_x,m)
    xzero = np.where(np.isnan(row_x),0,row_x)

    #train, dev, test dataset division
    x_train, x_dev, x_test = split_time_series_data(xzero)
    m_train, m_dev, m_test = split_time_series_data(m)
    d_train, d_dev, d_test = split_time_series_data(d)
    xpie_train, xpie_dev, xpie_test = split_time_series_data(xpie)
    s_train, s_dev, s_test = split_time_series_data(row_s)
    t_train, t_dev, t_test = split_time_series_data(row_t)
    label_train, label_dev, label_test = split_time_series_data(label)
    truedata_train, truedata_dev, truedata_test = split_time_series_data(true_data)
    print("x:",x_train.shape)
    print("xpie:",xpie_train.shape)
    print("s:",s_train.shape)
    print("t",t_train.shape)
    print("label:",label.shape)

    train_dataset = [x_train, m_train, d_train, xpie_train, s_train, t_train, label_train]
    dev_dataset = [x_dev, m_dev, d_dev, xpie_dev, s_dev, t_dev, label_dev]
    test_dataset = [x_test, m_test, d_test, xpie_test, s_test, t_test, label_test]

    #compile model
    num_hiddens = 64
    input_size = 96
    model = create_grud_model(output_activation='sigmoid',num_hiddens=num_hiddens, input_size= input_size)
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.summary()
    #show weight
    print("weights:", len(model.weights))
    print("non-trainable weights:", len(model.non_trainable_weights))
    print("trainable_weights:", len(model.trainable_weights))
    count = count_parameters(model)
    print('number of parameters : ' , count)


    history = model.fit(train_dataset,
                        truedata_train,
                        batch_size=48, epochs=300, validation_data=(dev_dataset,truedata_dev), shuffle=False)
    predictions = model.predict(test_dataset)
    #predictions = model.predict(pvdata)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()

    mse = np.mean(np.square(predictions - truedata_test))
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - truedata_test))
    r2 = r_squared(truedata_test, predictions)
    # mse = np.mean(np.square(predictions - pvdata_normalized))
    # rmse = np.sqrt(mse)
    # mae = np.mean(np.abs(predictions - pvdata_normalized))
    # r2 = r_squared(pvdata_normalized, predictions)
    print("mse value:",mse)
    print("mae value:",mae)
    print("r2 value:",r2)

    numpy_array = predictions
    columns = [f"col_{i}" for i in range(numpy_array.shape[1])]
    df = pd.DataFrame(numpy_array, columns=columns)
    df.to_csv((result_path))

    #record result
    file = open(result_record_path, 'a')
    file.write('GRUD_encoder_decoder_traindata10%,point misss,miss rate:{}\n'.format(str(int(miss_rate * 100))))
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    file.write('current time:{}'.format(formatted_time))
    file.write('\n')
    file.write('RMSE = {:.5f}   '.format(rmse))
    file.write('MAE = {:.5f}   '.format(mae))
    file.write('R2 = {:.5f}   '.format(r2))
    file.write('\n')
    file.close()
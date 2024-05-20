import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def count_parameters(model):
    """Count the number of trainable parameters in a TensorFlow model."""
    return sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

def split_time_series_data(data, train_size=0.1, val_size=0.03):
    total_size = len(data)
    train_end = int(total_size * train_size)
    val_end = train_end + int(total_size * val_size)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def r_squared(y_true, y_pred):
    # 计算总平方和
    total_variance = np.sum((y_true - np.mean(y_true)) ** 2)

    # 计算残差平方和
    residual_variance = np.sum((y_true - y_pred) ** 2)

    # 计算 R-squared
    r2 = 1 - (residual_variance / total_variance)

    return r2
def find_previous_observation(x, m):
    # 将输入的 TensorFlow 张量转换为 NumPy 数组
    x_np = np.array(x)
    m_np = np.array(m)

    # 获取输入 NumPy 数组的形状
    rows, cols = x_np.shape

    # 初始化结果数组，确保大小与输入 NumPy 数组 x 相同
    previous_observation = np.zeros_like(x_np, dtype=x_np.dtype)

    previous_value = None

    for i in range(rows):
        for j in range(cols):
            if m_np[i, j] == 1:
                # 当前位置不为空值，更新 previous_value
                previous_value = x_np[i, j]
            if previous_value is not None:
                # 将上一个非空值赋给当前位置
                previous_observation[i, j] = previous_value
            else:
                previous_observation[i, j] = 0

    return previous_observation

def create_time_interval(mask):
  interval = np.zeros_like(mask, dtype=int)
  rows, cols = mask.shape
  for j in range(cols):
    for i in range(rows):
      if i==0:
        interval[i,j] = 0
      elif mask[i,j]==1:
        interval[i,j] = 1
      elif mask[i,j]==0:
        n = i-1
        while n>=0 and mask[n,j]==0:
          n -= 1
        interval[i,j] = i-n

  return interval

def create_mask(array):
    # Create a mask with the same shape as the dataframe, initialized with ones.
    mask = np.ones_like(array, dtype=int)
    
    # Find the positions where the dataframe values are null and update the mask.
    mask = np.isnan(array)
    binary_mask = np.where(mask,0,1)
    
    return binary_mask


def df_norm(df):
    columns = df.columns.tolist()
    df_n = df.copy()
    for col in columns:
        mean = df_n[col].mean()
        std = df_n[col].std()
        df_n[col] = (df_n[col] - mean) / std
    return (df_n)


def normalize_data(data):
    nan_mask = np.isnan(data)
    
    if np.any(nan_mask):
        min_value = np.nanmin(data)
        max_value = np.nanmax(data)
        
        normalized_data = (data - min_value) / (max_value - min_value)
        normalized_data[nan_mask] = np.nan
    else:
        # 如果没有空值，直接进行标准化
        min_value = np.min(data)
        max_value = np.max(data)
        normalized_data = (data - min_value) / (max_value - min_value)
    
    return normalized_data, min_value, max_value

def denormalize_data(normalized_data, min_value, max_value):
    """
    将标准化后的数据还原为原先数据
    参数:
    - normalized_data: 标准化后的数组
    - min_value: 数据中的最小值
    - max_value: 数据中的最大值
    返回:
    - original_data: 还原后的数组
    """
    nan_mask = np.isnan(normalized_data)
    if np.isnan(min_value) or np.isnan(max_value):
        # 处理数据中全是NaN的情况
        original_data = normalized_data
    else:
        original_data = np.where(nan_mask, normalized_data, normalized_data * (max_value - min_value) + min_value)

    return original_data

def evaluate_and_plot(model, data):
    reconstructions = []

    data_mask = np.nan_to_num(data)
    #testdata_mask = np.squeeze(testdata_mask[0, :, :])
    data = tf.squeeze(data_mask)
    reconstructed_data = model(data)
    reconstructions.append(reconstructed_data.numpy())

    reconstructions = np.concatenate(reconstructions, axis=0)

    # Plot the original and reconstructed data
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Data')
    plt.plot(data[0], label='Original')
    plt.subplot(1, 2, 2)
    plt.title('Reconstructed Data')
    plt.plot(reconstructed_data[0], label='Reconstructed')

    plt.show()

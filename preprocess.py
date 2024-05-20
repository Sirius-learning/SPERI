import pandas as pd
import numpy as np
import os
from dataProcess import df_norm

current_path = os.getcwd()
#data_path=os.path.join(current_path,"data","SDU_stl.csv")
row_data_path = os.path.join(current_path,"data","SDU_PVdata_row.csv")
residual_data_path = os.path.join(current_path,"data","SDU_GRU_pre.csv")
label_data_path=os.path.join(current_path,"data","label_data.csv")
temp_path=os.path.join(current_path,"data","check.csv")
result_data_path=os.path.join(current_path,"data","resultdata.csv")
normalize_data_path=os.path.join(current_path,"data","SDU_PVdata_normalized.csv")
seasonal_trend_data_path = os.path.join(current_path,"data","seasonal_trend.csv")
seasonal_data_path = os.path.join(current_path,"data","seasonal_data.csv")
trend_data_path = os.path.join(current_path,"data","trend_data.csv")
mask_data_path = os.path.join(current_path,"data","SDU_PVdata_masked.csv")
# data=pd.read_csv(row_data_path)
# cols=["irradiance","realPower"]
# data_normalize=data[cols]
# data_normalize=df_norm(data_normalize)
# data_normalize.insert(0,'date',data['date'])
# data_normalize.to_csv(normalize_data_path)

# target_column = 'realPower'
# data = pd.read_csv(normalize_data_path)
# #设定空缺
# miss_rate = 0.2
# num_missing = int(len(data[target_column]) * miss_rate)
# missing_indices = np.random.choice(data.index, size=num_missing, replace=False)
# data.loc[missing_indices, target_column] = np.nan
# data.to_csv(mask_data_path)

#批量处理文件
# mask_rate = 0.3
# cols=['irradiance','realPower','realPower_Seasonal','realPower_Trend','realPower_Residual']
# for i in range(0,6):
#     miss_rate = mask_rate + 0.1 * i
#     data_path = os.path.join(current_path,"data","SDU_stl_" +str(int(miss_rate*100))+ "%.csv")
#     seasonal_data_path = os.path.join(current_path, "data", "seasonal_data_"+str(int(miss_rate*100))+ "%.csv")
#     trend_data_path = os.path.join(current_path, "data", "trend_data_"+str(int(miss_rate*100))+ "%.csv")
#     residual_data_path = os.path.join(current_path, "data", "SDU_GRU_pre_"+str(int(miss_rate*100))+ "%.csv")
#     data = pd.read_csv(data_path)
#     #print(data.dtypes)
#     seasonal_data = data[cols[2]].values.reshape(731,96)
#     trend_data = data[cols[3]].values.reshape(731,96)
#     residual_data = data[cols[4]].values.reshape(731,96)
#     label_data = data[cols[0]].values.reshape(731,96)
#     np.savetxt(seasonal_data_path,seasonal_data,delimiter=',')
#     np.savetxt(trend_data_path,trend_data,delimiter=',')
#     np.savetxt(residual_data_path,residual_data,delimiter=',')
#     np.savetxt(label_data_path,label_data,delimiter=',')

cols=['irradiance','realPower','realPower_Seasonal','realPower_Trend','realPower_Residual']
data_path = os.path.join(current_path,"data","SDU_PVdata_normalized.csv")
data = pd.read_csv(data_path)
label_data = data[cols[0]].values.reshape(731,96)
np.savetxt(label_data_path,label_data,delimiter=',')


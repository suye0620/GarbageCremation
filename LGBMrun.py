from sklearn.preprocessing import MinMaxScaler
from utils.datatools.dataSplit import dataSplit
from utils.datatools.LGBMTraining import LGBMTraining
from itertools import product
from tqdm import tqdm
import pandas as pd

# ###############################
# 数据集
# ###############################

# 读取
df_trainSet = pd.read_csv("./data/train/trainset.csv",encoding='utf-8-sig',index_col=['时间'],parse_dates=['时间'])

# 去除重复索引
df_trainSet = df_trainSet[~df_trainSet.index.duplicated()]
print(df_trainSet.index.shape)

# MMS标准化
scaler_steam = MinMaxScaler()
# fit_transform方法接受X-array的输入,所以列向量不行,要[[]]转成矩阵
seq_steam = scaler_steam.fit_transform(df_trainSet[["主蒸汽流量"]])
# 返回值会有多余维度,进行squeeze
seq_steam = seq_steam.squeeze()
(train_X,train_Y) , (val_X,val_Y) = dataSplit(seq_steam,lookback=7,random_state=43)

# ###############################
# LGBM训练
# ###############################
LGBMTraining(train_X,train_Y,val_X,val_Y,scaler_steam,random_state=43)

# ###############################
# 测试不同的lookback
# ############################### 
# list_lookbacks = [3,4,5,6,7,8,9,10,11]
# list_lookbacks = [12,13,14,15,16,17]
# list_randomseed = [11,22,33,44,55,66,77,88,99]
# for lookback,randomseed in tqdm(product(list_lookbacks,list_randomseed)):
#     (train_X,train_Y) , (val_X,val_Y) = dataSplit(seq_steam,lookback=lookback,random_state=randomseed)
#     LGBMTraining(train_X,train_Y,val_X,val_Y,scaler_steam,random_state=randomseed)

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def dataSplit(seq:pd.Series,lookback:int,valSet_ratio = 0.3, random_state = 42,shuffle=True):
    """
    Description
    -----------
    对待预测的时间序列进行切分,构造X->y数据集

    其中X的列数=lookback

    由于垃圾焚烧预测任务给定了测试集,故我们只划分训练集/验证集

    Parameters
    ----------
     seq: 序列
     lookback: 使用数目为lookback的时间步预测后面的序列值
     valSet_ratio: 划分验证集的比例
     random_state: 随机数种子
     shuffle: 默认为True,是否打乱数据集
    
    Returns
    -------
     (train_X,train_Y) , (val_X,val_Y): 划分的结果(np.array)
    """
    dataX, dataY = [], []
    for i in tqdm(range(len(seq) - lookback)):
        a = seq[i:(i + lookback)]
        dataX.append(a)
        dataY.append(seq[i + lookback])
    data_X,data_Y = np.array(dataX), np.array(dataY)
    
    # 去掉那个多的维度
    # data_X = data_X.squeeze()

    # 数据划分
    train_X,val_X , train_Y,val_Y = train_test_split(data_X,data_Y,test_size=valSet_ratio,random_state=random_state,shuffle=shuffle)
    print(train_X.shape)
    print(train_Y.shape)
    print(val_X.shape)
    print(val_Y.shape)
    val_size = val_X.shape[0]

    print("验证集时间步数为{}".format(val_size))
    return (train_X,train_Y) , (val_X,val_Y)
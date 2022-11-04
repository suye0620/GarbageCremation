from tqdm import tqdm
import pandas as pd
import numpy as np
def dataSplit(seq:pd.Series,lookback:int,trainSet_ratio = 0.7):
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
     trainSet_ratio: 划分训练集的比例
    
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
    train_size = int(len(data_X) * trainSet_ratio)
    val_size = len(data_X) - train_size

    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    val_X = data_X[train_size:train_size+val_size]
    val_Y = data_Y[train_size:train_size+val_size]

    print("验证集时间步数为{}".format(val_size))
    return (train_X,train_Y) , (val_X,val_Y)
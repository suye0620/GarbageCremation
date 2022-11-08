# 加载lightGBM模型
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def LGBMTraining(
    train_X: np.ndarray,
    train_Y: np.ndarray,
    val_X: np.ndarray,
    val_Y: np.ndarray,
    scaler_steam: MinMaxScaler,
    objective='regression',
    num_leaves=26, 
    learning_rate=0.05, 
    n_estimators=46,
    random_state=42
):

    """
    Description
    -----------
    对LGBM的训练进行封装,方便后续进行多次实验,确定最佳的lookback
    
    Parameters
    ----------
    train_X: 训练集X,
    train_Y: 训练集Y,
    val_X: 验证集X,
    val_Y: 验证集Y,
    scaler_steam: 对序列进行标准化的Scalar,
    other parameters: 来自lgb.LGBMRegressor()的参数,暂时放出这些参数用于调整
    
    Returns
    -------
    None

    """
    # 定义模型
    model = lgb.LGBMRegressor(objective=objective, num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators,random_state=random_state)
    
    # 训练/预测
    model.fit(train_X,train_Y)
    val_predict=model.predict(val_X)
    train_predict = model.predict(train_X)
    
    # 对预测值进行区间还原(之前做了scalar),再计算rmse的值
    val_Y = scaler_steam.inverse_transform(val_Y.reshape( len(val_Y) , 1 )).squeeze()
    val_predict = scaler_steam.inverse_transform(val_predict.reshape( len(val_predict) , 1 )).squeeze()
    train_predict = scaler_steam.inverse_transform(train_predict.reshape( len(train_predict) , 1 )).squeeze()
    
    # MSE
    mseval=mean_squared_error(val_Y,val_predict)
    maeval=mean_absolute_error(val_Y,val_predict)
    
    print("验证集MSELoss: ",mseval)
    print("验证集MAELoss: ",maeval)
    print("验证集RMSELoss: ",np.sqrt(mseval))
    
    # R2-score
    r2val = r2_score(val_Y,val_predict)
    print("验证集r2-score: ",r2val)
    (pd.DataFrame(
        data={
            # 虽然不知道为什么,但是下面时间这一列确实要用列表传递
            'time':[datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'lookback':train_X.shape[1],
            'model':'lightGBM',
            'random_state': random_state,
            'MSE_on_val':mseval,
            'RMSE_on_val':np.sqrt(mseval),
            'MAE_on_val':maeval,
            'r2score_on_val':r2val
        },
        ).to_csv("./log/lightGBMResults.csv",mode = 'a',encoding='utf-8-sig',index = False,header=False)
        )
    
    # 画出实际结果和预测的结果
    # y = val_Y
    # y_pre = val_predict
    # plt.plot(range(len(y)),y_pre,color = 'red',linewidth = 1.5,linestyle = '-.',label='prediction')
    # plt.plot(range(len(y)),y,color = 'blue',linewidth = 1.5,linestyle = '-', label='real')
    # plt.legend(loc='best')
    # plt.show()
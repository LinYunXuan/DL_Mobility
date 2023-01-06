import os

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第二块GPU（从0开始）
scaler=joblib.load('../Data/scalar')

def create_FNN(train_mobility_data):
    mobility_inputs = keras.Input(shape=train_mobility_data.shape[1:])
    flatten = layers.Flatten()
    x = flatten(mobility_inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs1 = layers.Dense(4)(x)
    outputs2 = layers.Dense(4)(x)
    outputs3 = layers.Dense(4)(x)
    outputs4 = layers.Dense(4)(x)
    outputs5 = layers.Dense(4)(x)
    outputs6 = layers.Dense(4)(x)
    model = keras.Model(inputs=mobility_inputs, outputs=[outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, ],
                        name='model')
    return model

def create_LSTM(train_mobility_data):
    # 构建mobility数据部分的网络
    mobility_inputs = keras.Input(shape=train_mobility_data.shape[1:])
    LSTM = layers.LSTM(units=256, return_sequences=True)
    x = LSTM(mobility_inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(units=128, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    mobility_outputs = layers.Dense(64, activation='relu')(x)

    # #构建policy数据部分的网络
    # policy_inputs = keras.Input(shape=train_policy_data.shape[1:])
    # LSTM=layers.LSTM(units=512, return_sequences=True)
    # x=LSTM(policy_inputs)
    # x = layers.Dropout(0.2)(x)
    # # x=layers.LSTM(units=256, return_sequences=True)(x)
    # # x = layers.Dropout(0.2)(x)
    # x=layers.Flatten()(x)
    # policy_outputs = layers.Dense(64, activation='relu')(x)

    # 合并后的网络
    # concat=layers.concatenate([mobility_outputs,policy_outputs])
    concat = mobility_outputs
    x1 = layers.Dense(32)(concat)
    x2 = layers.Dense(32)(concat)
    x3 = layers.Dense(32)(concat)
    x4 = layers.Dense(32)(concat)
    x5 = layers.Dense(32)(concat)
    x6 = layers.Dense(32)(concat)
    outputs1 = layers.Dense(4)(x1)
    outputs2 = layers.Dense(4)(x2)
    outputs3 = layers.Dense(4)(x3)
    outputs4 = layers.Dense(4)(x4)
    outputs5 = layers.Dense(4)(x5)
    outputs6 = layers.Dense(4)(x6)
    model = keras.Model(inputs=[mobility_inputs, ],
                        outputs=[outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, ], name='model')

    return model

def mae(pred,actual):
    pred=np.array(pred)
    actual=np.array(actual)
    pred=pred.swapaxes(0,1)
    pred=pred.swapaxes(1,2)
    # pred_un = scaler.inverse_transform(pred.reshape([-1,6]))
    # actual_un = scaler.inverse_transform(actual.reshape([-1,6]))
    pred=pred.reshape([-1,6])
    actual=actual.reshape([-1,6])
    mae_list = []
    for i in range(6):
        actual_data = actual[:, i]
        pred_data = pred[:, i]
        mae_unit = mean_absolute_error(actual_data, pred_data)

        mae_list.append(mae_unit)

    return mae_list

def rmse(pred, actual):
    '''
    :param pred:预测值
    :param actual: 真实值
    :return: rmse
    '''
    # scaler = joblib.load('../Data/scalar')
    pred=np.array(pred)
    actual=np.array(actual)
    pred=pred.swapaxes(0,1)
    pred=pred.swapaxes(1,2)
    # pred_un = scaler.inverse_transform(pred.reshape([-1,6]))
    # actual_un = scaler.inverse_transform(actual.reshape([-1,6]))
    pred=pred.reshape([-1,6])
    actual=actual.reshape([-1,6])
    mse_list=[]
    rmse_list=[]
    for i in range(6):
        actual_data=actual[:,i]
        pred_data=pred[:,i]
        mse_unit=mean_squared_error(actual_data,pred_data)
        rmse_unit=np.sqrt(mean_squared_error(actual_data, pred_data))
        mse_list.append(mse_unit)
        rmse_list.append(rmse_unit)

    return mse_list,rmse_list

train_mobility_data = np.load("../Data/train_mobility_data.npy")
train_policy_data = np.load("../Data/train_policy_data.npy")
train_label = np.load("../Data/train_label.npy")
test_mobility_data = np.load("../Data/test_mobility_data.npy")
test_policy_data = np.load("../Data/test_policy_data.npy")
test_label = np.load("../Data/test_label.npy")

# model=create_FNN(train_mobility_data)
# model.summary()
# model.load_weights(r'../logs/logFNN/best_model_FNN.hdf5')
model = tf.keras.models.load_model(r'../logs/logLSTM3/best_model_lstm2022-12-11 13:40:34.hdf5')
train_predict=model.predict([train_mobility_data,train_policy_data])
test_predict=model.predict([test_mobility_data,test_policy_data])
# train_predict=model.predict([train_mobility_data,])
# test_predict=model.predict([test_mobility_data,])
train_mse,train_rmse = rmse(train_predict, train_label)
test_mse,test_rmse = rmse(test_predict, test_label)
test_mae=mae(test_predict, test_label)
print(train_mse)
print(train_rmse)
print(test_mse)
print(test_rmse)
print(test_mae)

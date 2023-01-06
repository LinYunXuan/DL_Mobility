import os

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, r2_score
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

def rmse(pred, actual):
    '''
    :param pred:预测值
    :param actual: 真实值
    :return: rmse
    '''
    pred=np.array(pred)
    actual=np.array(actual)
    pred=pred.swapaxes(0,1)
    pred=pred.swapaxes(1,2)
    pred_un = scaler.inverse_transform(pred.reshape([-1,6]))
    print(pred_un)
    actual_un = scaler.inverse_transform(actual.reshape([-1,6]))
    print(actual_un)
    mse_list=[]
    rmse_list=[]
    print(actual_un.shape)
    for i in range(6):
        actual_data=actual_un[:,i]
        pred_data=pred_un[:,i]
        mse_unit=mean_squared_error(actual_data,pred_data)
        rmse_unit=np.sqrt(mean_squared_error(actual_data, pred_data))
        mse_list.append(mse_unit)
        rmse_list.append(rmse_unit)

    return mse_list,rmse_list

test_mobility_data = np.load("../Data/test_dataset/VNM_mobility_data.npy")
test_policy_data = np.load("../Data/test_dataset/VNM_policy_data.npy")
test_label = np.load("../Data/test_dataset/VNM_label.npy")

# test_mobility_data = np.load("../Data/test_mobility_data.npy")
# test_policy_data = np.load("../Data/test_policy_data.npy")
# test_label = np.load("../Data/test_label.npy")
print(test_mobility_data.shape)
print(test_mobility_data)
print(test_policy_data.shape)
print(test_label.shape)
# model=create_FNN(train_mobility_data)
# model.summary()
# model.load_weights(r'../logs/logFNN/best_model_FNN.hdf5')
model = tf.keras.models.load_model(r'../logs/logLSTM3/best_model_lstm2022-12-08 14:08:07.hdf5')

test_predict=model.predict([test_mobility_data,test_policy_data])
test_mse,test_rmse = rmse(test_predict, test_label)
print(test_mse)
print(test_rmse)
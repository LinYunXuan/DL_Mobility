import os

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow import keras
import numpy as np
import joblib
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import datetime

from tensorflow.python.keras import regularizers

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第二块GPU（从0开始）
#加载数据
train_mobility_data = np.load("../Data/train_mobility_data.npy")
train_policy_data = np.load("../Data/train_policy_data.npy")
train_label = np.load("../Data/train_label.npy")
test_mobility_data = np.load("../Data/test_mobility_data.npy")
test_policy_data = np.load("../Data/test_policy_data.npy")
test_label = np.load("../Data/test_label.npy")
print(train_mobility_data.shape)
print(train_policy_data.shape)
print(train_label.shape)
print('-'*200)



#构建mobility数据部分的网络
mobility_inputs = keras.Input(shape=train_mobility_data.shape[1:])
LSTM=layers.LSTM(units=256, return_sequences=True)
x=LSTM(mobility_inputs)
x = layers.Dropout(0.2)(x)
x=layers.LSTM(units=256,return_sequences=True)(x)
x = layers.Dropout(0.2)(x)
x=layers.LSTM(units=128)(x)
# x = layers.Dropout(0.2)(x)
x=layers.Flatten()(x)
mobility_outputs=layers.Dense(64, activation='relu')(x)
#构建policy数据部分的网络
policy_inputs = keras.Input(shape=train_policy_data.shape[1:])
LSTM=layers.LSTM(units=256, return_sequences=True)
x=LSTM(policy_inputs)
x = layers.Dropout(0.2)(x)
x=layers.LSTM(units=256,return_sequences=True)(x)
x = layers.Dropout(0.2)(x)
x=layers.LSTM(units=256)(x)
# x = layers.Dropout(0.2)(x)
# x=layers.LSTM(units=256, return_sequences=True)(x)
# x = layers.Dropout(0.2)(x)
# x=layers.LSTM(units=256, return_sequences=True)(x)
# x = layers.Dropout(0.2)(x)
x=layers.Flatten()(x)
policy_outputs = layers.Dense(64, activation='relu')(x)

#合并后的网络
concat=layers.concatenate([mobility_outputs,policy_outputs])
x = layers.Dense(256, activation='relu')(concat)
x = layers.Dense(64, activation='relu')(x)
outputs1 = layers.Dense(5,)(x)
outputs2 = layers.Dense(5,)(x)
outputs3 = layers.Dense(5,)(x)
outputs4 = layers.Dense(5,)(x)
outputs5 = layers.Dense(5,)(x)#kernel_regularizer=keras.regularizers.l2(0.001),bias_regularizer=keras.regularizers.l2(0.001)
outputs6 = layers.Dense(5,)(x)
model = keras.Model(inputs=[mobility_inputs,policy_inputs], outputs=[outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,], name='model')

#5.657502147037525, 9.619872600921221, 0.8910715734219103, 5.989992384693497, 35.46590601672676, 8.68958203524203
model.summary()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100,
    decay_rate=0.8,
    staircase=False,)
optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mse')
log_file = os.path.join('../logs/logLSTM3', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
tensorboard_callback = TensorBoard(log_file)
checkpoint_file = "../logs/logLSTM3/best_model_lstm"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+".hdf5"
earlyStopping_callback=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=5e-6,
    patience=40,
    verbose=0,
    mode="min",
    baseline=None,
    restore_best_weights=False,
)

checkpoint_callback = ModelCheckpoint(filepath=checkpoint_file,
                                      monitor='loss',
                                      mode='min',
                                      save_best_only=True,
                                      save_weights_only=False)

history = model.fit(x=[train_mobility_data,train_policy_data],
                    y=[train_label[:,:,0],train_label[:,:,1],train_label[:,:,2],train_label[:,:,3],train_label[:,:,4],train_label[:,:,5],],
                    batch_size=256,epochs=500,
                    callbacks=[tensorboard_callback, checkpoint_callback,earlyStopping_callback],
                    validation_split=0.25,
                    shuffle=True,
                    validation_batch_size=256,
                    validation_steps=20,
                    workers=32
                    )
plt.figure(figsize=(16, 8))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend(loc='best')
plt.show()

train_predict=model.predict([train_mobility_data,train_policy_data])
test_predict=model.predict([test_mobility_data,test_policy_data])

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
    scaler = joblib.load('../Data/scalar')
    pred=np.array(pred)
    actual=np.array(actual)
    pred=pred.swapaxes(0,1)
    pred=pred.swapaxes(1,2)
    pred_un = scaler.inverse_transform(pred.reshape([-1,6]))
    actual_un = scaler.inverse_transform(actual.reshape([-1,6]))
    # pred=pred.reshape([-1,6])
    # actual=actual.reshape([-1,6])
    mse_list=[]
    rmse_list=[]
    for i in range(6):
        actual_data=actual_un[:,i]
        pred_data=pred_un[:,i]
        mse_unit=mean_squared_error(actual_data,pred_data)
        rmse_unit=np.sqrt(mean_squared_error(actual_data, pred_data))
        mse_list.append(mse_unit)
        rmse_list.append(rmse_unit)

    return mse_list,rmse_list

train_mse,train_rmse = rmse(train_predict, train_label)
test_mse,test_rmse = rmse(test_predict, test_label)
test_mae=mae(test_predict, test_label)
# print(train_mse)
print('MSE:\n',test_mse)
# print(train_rmse)
# print(test_rmse)
print('MAE\n',test_mae)
# score = r2_score(test_label, test_predict)
#
# print("r^2 值为： ", score)
# plt.figure(figsize=(16,8))
# plt.plot(test_label[:], label="True value")
# plt.plot(test_predict[:], label="Pred value")
# plt.legend(loc='best')
# plt.show()

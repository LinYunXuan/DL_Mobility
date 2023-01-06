import os

from sklearn.metrics import r2_score, mean_squared_error
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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用第二块GPU（从0开始）
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

#训练集shuffle
np.random.seed(120)  # 设置随机种子，让每次结果都一样，方便对照
np.random.shuffle(train_mobility_data)  # 使用shuffle()方法，让输入x_train乱序
np.random.seed(120)  # 设置随机种子，让每次结果都一样，方便对照
np.random.shuffle(train_policy_data)  # 使用shuffle()方法，让输入x_train乱序
np.random.seed(120)  # 设置随机种子，让每次结果都一样，方便对照
np.random.shuffle(train_label)  # 使用shuffle()方法，让输入y_train乱序

# #归一化处理
# mobility_scaler = joblib.load('../Data/mobility_data_scalar')
# train_mobility_data=train_mobility_data.reshape([train_mobility_data.shape[0],-1])
# train_mobility_data = mobility_scaler.inverse_transform(train_mobility_data[:,:18])
# train_mobility_data=pd.DataFrame(train_mobility_data)
# print(train_mobility_data)
# train_mobility_data.to_csv('../test.csv')
# print('-'*200)


#构建mobility数据部分的网络
mobility_inputs = keras.Input(shape=train_mobility_data.shape[1:])
flatten = layers.Flatten()
x = flatten(mobility_inputs)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)

outputs1 = layers.Dense(4)(x)
outputs2 = layers.Dense(4)(x)
outputs3 = layers.Dense(4)(x)
outputs4 = layers.Dense(4)(x)
outputs5 = layers.Dense(4,kernel_regularizer=keras.regularizers.l2(0.0001))(x)
outputs6 = layers.Dense(4)(x)
model = keras.Model(inputs=mobility_inputs, outputs=[outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,], name='model')
model.summary()
#2022-12-08 21:16:44
#8.001661487844963, 11.787056607658734, 0.8560904075086334, 7.584137062216688, 143.42004296534498, 13.76750415643013

#8.349479518479054, 12.416507451740731, 0.8346437497445852, 7.272620434675367, 135.01002704879826, 13.384814509074909
lr=tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0005,
    decay_steps=50,
    decay_rate=0.9,
)
optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer='sgd', loss='mse')
log_file = os.path.join('../logs/logFNN', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
tensorboard_callback = TensorBoard(log_file)
checkpoint_file = "../logs/logFNN/best_model_FNN"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+".hdf5"
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_file,
                                      monitor='loss',
                                      mode='min',
                                      save_best_only=True,
                                      save_weights_only=False)
earlyStopping_callback=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=5e-6,
    patience=30,
    verbose=0,
    mode="min",
    baseline=None,
    restore_best_weights=False,
)

history = model.fit(x=train_mobility_data,
                    y=[train_label[:,:,0],train_label[:,:,1],train_label[:,:,2],train_label[:,:,3],train_label[:,:,4],train_label[:,:,5],],
                    batch_size=256,epochs=1000,
                    callbacks=[tensorboard_callback, checkpoint_callback,earlyStopping_callback],
                    validation_split=0.25,
                    shuffle=True,
                    validation_batch_size=256,
                    validation_steps=20,
                    )
plt.figure(figsize=(16, 8))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend(loc='best')
plt.show()

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
    mse_list=[]
    rmse_list=[]
    print(actual.shape)
    for i in range(6):
        actual_data=actual[:,i]
        pred_data=pred[:,i]
        mse_unit=mean_squared_error(actual_data,pred_data)
        rmse_unit=np.sqrt(mean_squared_error(actual_data, pred_data))
        mse_list.append(mse_unit)
        rmse_list.append(rmse_unit)

    return mse_list,rmse_list
train_predict=model.predict([train_mobility_data,])
test_predict=model.predict([test_mobility_data,])
train_mse,train_rmse = rmse(train_predict, train_label)
test_mse,test_rmse = rmse(test_predict, test_label)
print(train_mse)
print(test_mse)
print(train_rmse)
print(test_rmse)
# score = r2_score(test_label[:,:,0], test_predict)
# print("r^2 值为： ", score)

# plt.figure(figsize=(16,8))
# plt.plot(test_label, label="True value")
# plt.plot(test_predict, label="Pred value")
# plt.legend(loc='best')
# plt.show()
#归一化处理
#对test的国家分别进行predict
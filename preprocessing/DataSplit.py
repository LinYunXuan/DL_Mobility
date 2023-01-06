import joblib
import numpy as np
import os
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler

def remove_duplication(col):
    data = list()
    for i, v in enumerate(col):
        if isinstance(v, np.ndarray):
            # print(2)
            data.add(v[0])
            print(data)
        else:
            if str(v) in data:
                continue
            else:
                data.append(str(v))
    return data


# 将数据归一化处理
dataset = pd.read_csv('../Data/feature_data.csv')
mobility_data=pd.concat([dataset.iloc[:,1:3],dataset.iloc[:,4:22]],axis=1)
policy_data=pd.concat([dataset.iloc[:,1:3],dataset.iloc[:,22:55]],axis=1)
# print(policy_data)
mobility_scaler = MinMaxScaler(feature_range=[0, 1])
policy_scaler = MinMaxScaler(feature_range=[0, 1])
scaler=MinMaxScaler(feature_range=[0, 1])
scaler.fit(mobility_data.iloc[:,2:8])
mobility_scaler_data = pd.DataFrame(mobility_scaler.fit_transform(mobility_data.iloc[:,2:]))
policy_scaler_data = pd.DataFrame(policy_scaler.fit_transform(policy_data.iloc[:,2:]))
policy_scaler_data.to_csv('../Data/policy_scaler_data.csv')
mobility_scaler_data.to_csv('../Data/mobility_scaler_data.csv')
print(111)
scaler_filename="../Data/scalar"
mobility_scaler_filename = "../Data/mobility_data_scalar"
policy_scaler_filename = "../Data/policy_data_scalar"
# print(mobility_scaler_data)
with open(mobility_scaler_filename, 'w') as f:
    joblib.dump(mobility_scaler, mobility_scaler_filename)
with open(policy_scaler_filename, 'w') as f:
    joblib.dump(policy_scaler, policy_scaler_filename)
with open(scaler_filename, 'w') as f:
    joblib.dump(scaler, scaler_filename)

# 将数据按国家划分，使用map的形式存储
code = remove_duplication(dataset['Code'].values)
country_mobility_map = {}
country_policy_map = {}
for i in code:
    country_mobility_map.update({i: mobility_scaler_data.loc[dataset["Code"] == i]})
    country_policy_map.update({i: policy_scaler_data.loc[dataset["Code"] == i]})
# print(policy_scaler_data.loc[dataset["Code"] == 'ARE'])
# print(mobility_scaler_data.loc[dataset["Code"] == 'ARE'])
# mobility_scaler_data.to_csv('../Data/mobility_scaler_data.csv')
# 划分数据集
def split_sequence(mobility_sequence,policy_sequence, n_steps_in=10, n_steps_out=4):
    """
    :param sequence:list 2020-2-17至2022-8-1的dataset
    :param n_steps_in: input的数量
    :param n_steps_out: output的数量
    :return: input,output的序列
    """
    X,X2, y = [], [], []
    for i in range(len(mobility_sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(mobility_sequence):
            break
        # gather input and output parts of the pattern
        seq_x1,seq_x2, seq_y = mobility_sequence.iloc[i:end_ix, :],policy_sequence.iloc[i:end_ix, :], mobility_sequence.iloc[end_ix:out_end_ix,:6]
        # seq_y=scaler.inverse_transform(seq_y)
        # seq_y=np.array(seq_y).flatten()
        X.append(seq_x1)
        X2.append(seq_x2)
        y.append(seq_y)
    return np.array(X),np.array(X2), np.array(y)


def split_train_and_test(code, country_mobility_map ,country_policy_map):
    random.seed(1)
    number = len(code)  # 国家个数
    rate = 0.2  # 抽取的验证集的比例，占总数据的多少
    picknumber = int(number * rate)  # 按照rate比例从数据集中取数据做验证集
    print(picknumber)
    print(code)
    test_code = random.sample(code, picknumber)  # 随机选取需要数量的样本图片
    train_code = set(code) - set(test_code)
    train_code =list(train_code)
    print(test_code)
    # 处理test_data
    test_mobility_data, test_policy_data, test_label = split_sequence(
        mobility_sequence=country_mobility_map[test_code[0]],
        policy_sequence= country_policy_map[test_code[0]],n_steps_in=10, n_steps_out=5)
    for i in test_code[1:]:
        data1, data2, label = split_sequence(
            mobility_sequence=country_mobility_map[i],
            policy_sequence=country_policy_map[i], n_steps_in=10, n_steps_out=5)
        test_mobility_data = np.concatenate([test_mobility_data, data1], axis=0)
        test_policy_data = np.concatenate([test_policy_data, data2], axis=0)
        test_label = np.concatenate([test_label, label], axis=0)

    # 处理train_data
    train_mobility_data, train_policy_data, train_label = split_sequence(
        mobility_sequence=country_mobility_map[train_code[0]],
        policy_sequence=country_policy_map[train_code[0]], n_steps_in=10, n_steps_out=5)
    for i in train_code[1:]:
        data1, data2, label = split_sequence(
            mobility_sequence=country_mobility_map[i],
            policy_sequence=country_policy_map[i], n_steps_in=10, n_steps_out=5)
        train_mobility_data = np.concatenate([train_mobility_data, data1], axis=0)
        train_policy_data= np.concatenate([train_policy_data, data2], axis=0)
        train_label = np.concatenate([train_label, label], axis=0)
    return train_mobility_data,train_policy_data, train_label, test_mobility_data, test_policy_data, test_label


train_mobility_data,train_policy__data, train_label, test_mobility_data, test_policy_data, test_label = split_train_and_test(code,country_mobility_map ,country_policy_map)
print(train_mobility_data.shape)
print(train_policy__data.shape)
print(test_label.shape,)
print(train_label.shape,)

np.save("../Data/train_mobility_data.npy", train_mobility_data)
np.save("../Data/train_policy_data.npy", train_policy__data)
np.save("../Data/train_label.npy", train_label)
np.save("../Data/test_mobility_data.npy", test_mobility_data)
np.save("../Data/test_policy_data.npy", test_policy_data)
np.save("../Data/test_label.npy", test_label)



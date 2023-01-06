import joblib
import pandas as pd
import numpy as np

dataset1 = pd.read_csv(r"../Data/OxCGRT_nat_latest.csv", )
dataset1['Date'] = pd.to_datetime(dataset1['Date'], format='%Y/%m/%d', errors='ignore')
dataset1 = dataset1.loc[(dataset1['Date'] <= '2022/08/01') & (dataset1['Date'] >= '2020/02/17')
                        & (dataset1['C1M_School closing'].notna()) & (dataset1['ConfirmedCases'].notna())]

dataset2 = pd.read_csv(r"../Data/changes-visitors-covid.csv")
dataset2.replace('', np.nan, inplace=True)
dataset2 = dataset2.dropna()
dataset2['Day'] = pd.to_datetime(dataset2['Day'],format='%Y/%m/%d')
dataset2 = dataset2.loc[(dataset2['Day'] <= '2022/08/01') & (dataset2['Day'] >= '2020/02/17')]


policy = dataset1.loc[dataset1['CountryCode']=='VNM']
policy.reset_index(drop=True,inplace=True)
policy.rename(columns={'CountryCode':'Code','Date':'Day'},inplace=True)
policy.set_index(['Code','Day'],inplace=True)
mobility = dataset2.loc[dataset2['Code']=='VNM']
mobility.reset_index(drop=True,inplace=True)
mobility.set_index(['Code','Day'],inplace=True)
policy=policy.drop(axis=1,columns=['CountryName'])

fusion_data=pd.concat([mobility,policy],axis=1,join='inner')
fusion_data.reset_index(inplace=True)
# fusion_data.insert(0, 'Code', index_ar)
mobility_data = fusion_data.iloc[:, 3:9]
mobility_average_data = mobility_data.rolling(5).mean()
# 前四天的数据处理

mobility_average_data.iloc[0, :] = mobility_data.iloc[0, :]
mobility_average_data.iloc[1, :] = mobility_data.iloc[: 2, :].mean()
mobility_average_data.iloc[2, :] = mobility_data.iloc[:3, :].mean()
mobility_average_data.iloc[ 3, :] = mobility_data.iloc[:4, :].mean()
mobility_col_name = mobility_average_data.columns.values
average_col_map = {}
for i in mobility_col_name:
    average_col_map.update({i: i + '_average'})
mobility_average_data.rename(columns=average_col_map, inplace=True)
for index, value in enumerate(mobility_average_data.columns.values):
    fusion_data.insert(9 + index, value, mobility_average_data.iloc[:, index])

# an average computed from the last date of update of any control measure.
policy_data = pd.concat([fusion_data.iloc[:, 15:23], fusion_data.iloc[:, [25, 26, 27, 30]]], axis=1)
# 获取每一个措施发生改变的时间序列policy_change_flag_data,以及其汇总policy_gather_flag
policy_col_name = policy_data.columns.values
policy_change_flag_data = policy_data.copy()
policy_gather_flag = None
for index, value in enumerate(policy_change_flag_data.columns.values):
    col = policy_change_flag_data[value]
    data = col.ne(col.shift())
    data[0] = False
    if index == 0:
        policy_gather_flag = data
    else:
        policy_gather_flag = -((-policy_gather_flag) & (-data))
    policy_change_flag_data.insert(12 + index, value + '_flag', data.astype(int))
policy_change_flag_data.insert(24, 'policy_gather_flag', policy_gather_flag.astype(int))
policy_change_flag_data = policy_change_flag_data.drop(list(policy_change_flag_data)[0:12], axis=1)
for index, value in enumerate(policy_change_flag_data.columns.values):
    fusion_data.insert(15 + index, value, policy_change_flag_data.iloc[:, index])

# 获取各国政策变化的时间点
policy_change_index = []
before = 0
counter = 1
# 记录政策发生变化的时间点和每个国家的当前时间点policy_change_index
for index, value in zip(policy_gather_flag.index, policy_gather_flag):
    if value:
        policy_change_index.append(index)
# 求an average computed from the last date of update of any control measure
mobility_last_average_data = mobility_data.copy()
average_col_map = {}
for i in mobility_col_name:
    average_col_map.update({i: i + '_last_average'})
mobility_last_average_data.rename(columns=average_col_map, inplace=True)
for index, value in enumerate(policy_change_index):
    if index == len(policy_change_index) - 1:
        break
    mobility_last_average_data.iloc[before:value, :] = mobility_last_average_data.iloc[before:value, :].mean()
    before = value
for index, value in enumerate(mobility_last_average_data.columns.values):
    fusion_data.insert(15 + index, value, mobility_last_average_data.iloc[:, index])

#按时间序列划分数据
mobility_data=pd.concat([fusion_data.iloc[:,1:3],fusion_data.iloc[:,3:21]],axis=1)
policy_data=pd.concat([fusion_data.iloc[:,1:3],fusion_data.iloc[:,21:54]],axis=1)
print(policy_data)
scaler_filename="../Data/scalar"
mobility_scaler_filename = "../Data/mobility_data_scalar"
policy_scaler_filename = "../Data/policy_data_scalar"
mobility_scaler = joblib.load(mobility_scaler_filename)
policy_scaler = joblib.load(policy_scaler_filename)
mobility_scaler_data = pd.DataFrame(mobility_scaler.transform(mobility_data.iloc[:,2:]))
policy_scaler_data = pd.DataFrame(policy_scaler.transform(policy_data.iloc[:,2:]))
# mobility_scaler_data.to_csv("a.csv")
print(policy_scaler_data)
print(mobility_scaler_data)
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
        seq_x1,seq_x2, seq_y = mobility_sequence.iloc[i:end_ix, :],policy_sequence.iloc[i:out_end_ix, :], mobility_sequence.iloc[end_ix:out_end_ix,:6]
        # seq_y=np.array(seq_y).flatten()
        X.append(seq_x1)
        X2.append(seq_x2)
        y.append(seq_y)
    return np.array(X),np.array(X2), np.array(y)


def split_train_and_test( country_mobility_map ,country_policy_map):
    # 处理test_data
    test_mobility_data, test_policy_data, test_label = split_sequence(
        mobility_sequence=country_mobility_map,
        policy_sequence= country_policy_map,n_steps_in=10, n_steps_out=4)
    return test_mobility_data, test_policy_data, test_label

test_mobility_data, test_policy_data, test_label = split_train_and_test(mobility_scaler_data ,policy_scaler_data)
print(test_mobility_data.shape)
print(test_policy_data.shape)
print(test_label.shape,)


np.save("../Data/test_dataset/VNM_mobility_data.npy", test_mobility_data)
np.save("../Data/test_dataset/VNM_policy_data.npy", test_policy_data)
np.save("../Data/test_dataset/VNM_label.npy", test_label)


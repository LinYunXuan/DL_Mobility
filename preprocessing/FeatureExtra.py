import numpy as np
import pandas as pd

def remove_duplication(col):
    data = set()
    for i, v in enumerate(col):
        if isinstance(v, np.ndarray):
            # print(2)
            data.add(v[0])
        else:
            # print(1)
            data.add(str(v))
    return data


dataset = pd.read_csv(f"../Data/fusionData.csv", )
# extra feature

# mobility average
mobility_data = dataset.iloc[:, 3:9]
mobility_average_data = mobility_data.rolling(5).mean()
# 前四天的数据处理
day_len = 897  # pd.to_datetime('2022-8-1')-pd.to_datetime('2020-2-17').一个国家记录的天数
code_set = remove_duplication(dataset[['Code']].values)

for i in range(len(code_set)):
    mobility_average_data.iloc[i * day_len, :] = mobility_data.iloc[i * day_len, :]
    mobility_average_data.iloc[i * day_len + 1, :] = mobility_data.iloc[i * day_len:i * day_len + 2, :].mean()
    mobility_average_data.iloc[i * day_len + 2, :] = mobility_data.iloc[i * day_len:i * day_len + 3, :].mean()
    mobility_average_data.iloc[i * day_len + 3, :] = mobility_data.iloc[i * day_len:i * day_len + 4, :].mean()
mobility_col_name = mobility_average_data.columns.values
average_col_map = {}
for i in mobility_col_name:
    average_col_map.update({i: i + '_average'})
mobility_average_data.rename(columns=average_col_map, inplace=True)
for index, value in enumerate(mobility_average_data.columns.values):
    dataset.insert(9 + index, value, mobility_average_data.iloc[:, index])

    # an average computed from the last date of update of any control measure.
policy_data = pd.concat([dataset.iloc[:, 15:23], dataset.iloc[:, [25, 26, 27, 30]]], axis=1)
# 获取每一个措施发生改变的时间序列policy_change_flag_data,以及其汇总policy_gather_flag
policy_col_name = policy_data.columns.values
policy_change_flag_data = policy_data.copy()
policy_gather_flag = None
for index, value in enumerate(policy_change_flag_data.columns.values):
    col = policy_change_flag_data[value]
    data = col.ne(col.shift())
    for i in range(len(col)) :
        if i%day_len ==0:
            print(i)
            data[i]=False
    data[0] = False
    if index == 0:
        policy_gather_flag = data
    else:
        policy_gather_flag = -((-policy_gather_flag) & (-data))
    policy_change_flag_data.insert(12 + index, value + '_flag', data.astype(int))
policy_change_flag_data.insert(24, 'policy_gather_flag', policy_gather_flag.astype(int))
policy_change_flag_data = policy_change_flag_data.drop(list(policy_change_flag_data)[0:12], axis=1)
for index, value in enumerate(policy_change_flag_data.columns.values):
    dataset.insert(15 + index, value, policy_change_flag_data.iloc[:, index])

# 获取各国政策变化的时间点
policy_change_index = []
before = 0
country_count = len(mobility_data) / day_len
counter = 1
# 记录政策发生变化的时间点和每个国家的当前时间点policy_change_index
for index, value in zip(policy_gather_flag.index, policy_gather_flag):
    if value:
        policy_change_index.append(index)
    elif index == counter * day_len:#每个国家的数据都需要单独计算，因此需要将每个国家的第一天作为政策更改的日期
        policy_change_index.append(day_len * counter)
        counter += 1
policy_change_index.append(len(policy_gather_flag))
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
    dataset.insert(15 + index, value, mobility_last_average_data.iloc[:, index])
dataset.to_csv("../Data/feature_data.csv")

exit()

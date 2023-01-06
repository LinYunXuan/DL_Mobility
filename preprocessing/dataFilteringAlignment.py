import pandas as pd
import numpy as np


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


def save_Na(dataset):
    x = dataset[dataset[dataset.columns].isna().any(axis=1)]
    # print(x,type(x))
    x.to_csv(r'../Data/na.csv')


dataset1 = pd.read_csv(r"../Data/OxCGRT_nat_latest.csv", )
dataset1['Date'] = pd.to_datetime(dataset1['Date'], format='%Y/%m/%d', errors='ignore')
dataset1['H4_Emergency investment in healthcare'].replace('', 0, inplace=True)
dataset1.replace(' ', np.nan, inplace=True)
dataset1 = dataset1.loc[(dataset1['Date'] <= '2022/08/01') & (dataset1['Date'] >= '2020/02/17')
                        & (dataset1['C1M_School closing'].notna()) & (dataset1['ConfirmedCases'].notna())]
# dataset1=dataset1.dropna()

dataset2 = pd.read_csv(r"../Data/changes-visitors-covid.csv")
dataset2.replace('', np.nan, inplace=True)
dataset2 = dataset2.dropna()
dataset2['Day'] = pd.to_datetime(dataset2['Day'],format='%Y/%m/%d')
dataset2 = dataset2.loc[(dataset2['Day'] <= '2022/08/01') & (dataset2['Day'] >= '2020/02/17')]
code_counter = dataset2['Code'].value_counts()
s = set(code_counter.index) - set(code_counter[code_counter == 897].index)
# mes = dataset1[['MEASURE']].values
# measures = remove_duplication(mes)
ISO_policy = remove_duplication(dataset1[['CountryCode']].values)
ISO_mobility = remove_duplication(dataset2['Code'].values)

jiaoji = ISO_mobility & ISO_policy - s

policy = dataset1.loc[dataset1['CountryCode'].isin(jiaoji)]
policy.reset_index(drop=True,inplace=True)
policy.set_index(['CountryCode','Date'],inplace=True)
mobility = dataset2.loc[dataset2['Code'].isin(jiaoji)]
mobility.reset_index(drop=True,inplace=True)
mobility.set_index(['Code','Day'],inplace=True)
policy=policy.drop(axis=1,columns=['CountryName'])
# ISO_policy = remove_duplication(policy[['CountryCode']].values)
# ISO_mobility = remove_duplication(mobility['Code'].values)
# print(len(ISO_policy), len(ISO_mobility), len(jiaoji))
# print(ISO_mobility,'\n',ISO_policy)
# print(jiaoji)
# print(policy, len(policy))
# print(mobility, len(mobility))
fusion_data=pd.concat([mobility,policy],axis=1)
print(fusion_data)
fusion_data.to_csv('../Data/fusionData.csv',index_label=['Code','Date'])
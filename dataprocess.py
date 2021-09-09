import pandas as pd
import numpy as np
import matplotlib as plt


df_accessory_1 = pd.read_excel("../2021/B/附件1.xlsx")
df_accessory_2 = pd.read_excel("../2021/B/附件2.xlsx", header=1)

# 处理附件1
saved_index = 0
for index, _ in df_accessory_1.iterrows():
    if pd.isnull(df_accessory_1.iloc[index, 0]):
        df_accessory_1.iloc[index, 0] = df_accessory_1.iloc[saved_index, 0]
        df_accessory_1.iloc[index, 1] = df_accessory_1.iloc[saved_index, 1]
    else:
        saved_index = index

# 处理附件2
df_accessory_2.drop(labels=0, axis=0, inplace=True)
df_accessory_2 = df_accessory_2.reset_index(drop=True)
columns = {"选择性(%)": "乙烯选择性(%)",
           "Unnamed: 3": "C4烯烃选择性(%)",
           "Unnamed: 4": "乙醛选择性(%)",
           "Unnamed: 5": "碳数为4-12脂肪醇(%)",
           "Unnamed: 6": "甲基苯甲醛和甲基苯甲醇(%)",
           "Unnamed: 7": "其他"}
df_accessory_2.rename(columns=columns, inplace=True)

# 处理两附件的数据类型
df_accessory_1.iloc[:, 2:] = df_accessory_1.iloc[:, 2:].astype(np.float64)
df_accessory_2 = df_accessory_2.astype(np.float64)

df_accessory_1.to_csv('./accessory1.csv', sep=',', header=True, index=False)
df_accessory_2.to_csv('./accessory2.csv', sep=',', header=True, index=False)
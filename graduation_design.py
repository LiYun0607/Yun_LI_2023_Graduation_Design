import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib


sns.set_theme(style="darkgrid")


# 平滑处理，类似tensorboard的smoothing函数。
def smooth(read_path, save_path, file_name, x='Number of Iterations', y='Reward', weight=0.1):

    data = pd.read_csv(read_path + file_name)
    scalar = data[y].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({x: data[x].values, y: smoothed})
    save.to_csv(save_path + 'smooth_'+ file_name)


# 平滑预处理原始reward数据
smooth(read_path='./data/', save_path='./data/', file_name='lost_data_dqn1.csv')
smooth(read_path='./data/', save_path='./data/', file_name='lost_data_dqn2.csv')
smooth(read_path='./data/', save_path='./data/', file_name='PPO_data_0.csv')
smooth(read_path='./data/', save_path='./data/', file_name='PPO_data_1.csv')
smooth(read_path='./data/', save_path='./data/', file_name='lost_data_cpo.csv')
smooth(read_path='./data/', save_path='./data/', file_name='lost_data_cpo2.csv')
smooth(read_path='./data/', save_path='./data/', file_name='CNN_data1.csv')
smooth(read_path='./data/', save_path='./data/', file_name='CNN_data2.csv')
smooth(read_path='./data/', save_path='./data/', file_name='lost_data_random1.csv')
smooth(read_path='./data/', save_path='./data/', file_name='lost_data_random2.csv')


df1 = pd.read_csv('./data/smooth_PPO_data_0.csv')
df2 = pd.read_csv('./data/smooth_PPO_data_1.csv')
# df3 = pd.read_csv('./data/PPO_data_2.csv')
# sns.lineplot(x="number of iteration", y="lost data all", data=data)
df4 = df1.append(df2)
# df5 = df4.append(df3)
df4.index = range(len(df4))

df6 = pd.read_csv('./data/smooth_lost_data_cpo.csv')
df7 = pd.read_csv('./data/smooth_lost_data_cpo2.csv')
df8 = df6.append(df7)
df8.index = range(len(df8))

df9 = pd.read_csv('./data/smooth_lost_data_dqn1.csv')
df10 = pd.read_csv('./data/smooth_lost_data_dqn2.csv')
df11 = df9.append(df10)
df11.index = range(len(df11))

df12 = pd.read_csv('./data/smooth_CNN_data1.csv')
df13 = pd.read_csv('./data/smooth_CNN_data2.csv')
df14 = df12.append(df13)
df14.index = range(len(df14))

df15 = pd.read_csv("data/smooth_lost_data_random1.csv")
df16 = pd.read_csv("data/smooth_lost_data_random2.csv")
df17 = df15.append(df16)
df17.index = range(len(df17))
# fig, ax = plt.subplots()
# # sns.lineplot(x="Number of Iterations", y="Reward", data=df11, label='MA-DQN', color='chocolate')
# # sns.lineplot(x="Number of Iterations", y="Reward", data=df22, label='MA-DQN_decay_lr', color='violet')
# # sns.lineplot(x="Number of Iterations", y="Reward", data=df33, label='MA-PPO', color='b')
# # sns.lineplot(x="Number of Iterations", y="Reward", data=df44, label='Random', color='g')
# # fig.savefig('Reward.pdf', format='pdf', dpi=1200)
#
# sns.lineplot(x="Number of Iterations", y="Lost Data", data=df8, label='MA-Transformer-PPO', color='chocolate', alpha=0.5)
# # sns.lineplot(x="Number of Iterations", y="Lost Data", data=df11, label='MA-DQN', color='violet')
# sns.lineplot(x="Number of Iterations", y="Lost Data", data=df5, label='MA-PPO', color='b', alpha=0.5)
# # sns.lineplot(x="Number of Iterations", y="Lost Data", data=df44, label='Random', color='g')
# fig.savefig('Lost.pdf', format='pdf', dpi=1200)
# plt.show()
fig, ax = plt.subplots()
plt.ylim(-250, -50)
# plt.ylim(0, 500)

# sns.lineplot(x="Number of Iterations", y="Lost Data", data=df17, label='Random', color='violet')
# sns.lineplot(x="Number of Iterations", y="Lost Data", data=df11, label='MA-DQN', color='chocolate')
# sns.lineplot(x="Number of Iterations", y="Lost Data", data=df4, label='SPMA-PPO', color='b')
# sns.lineplot(x="Number of Iterations", y="Lost Data", data=df14, label='CNN-SPMA-PPO', color='g')
# sns.lineplot(x="Number of Iterations", y="Lost Data", data=df8, label='T-SPMA-PPO', color='darkred')

sns.lineplot(x="Number of Iterations", y="Reward", data=df17, label='Random', color='violet')
sns.lineplot(x="Number of Iterations", y="Reward", data=df11, label='MA-DQN', color='chocolate')
sns.lineplot(x="Number of Iterations", y="Reward", data=df4, label='SPMA-PPO', color='b')
sns.lineplot(x="Number of Iterations", y="Reward", data=df14, label='CNN-SPMA-PPO', color='g')
sns.lineplot(x="Number of Iterations", y="Reward", data=df8, label='T-SPMA-PPO', color='darkred')


fig.savefig('reward.pdf', format='pdf', dpi=1200)
plt.show()

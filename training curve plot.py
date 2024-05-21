import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib


sns.set_theme(style="darkgrid")


# 平滑处理，类似tensorboard的smoothing函数。
def smooth(read_path, save_path, file_name, x='number of iteration', y='lost data all', weight=0.9):

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
smooth(read_path='./data/', save_path='./data/', file_name='Transformer1.csv')
smooth(read_path='./data/', save_path='./data/', file_name='Transformer2.csv')
smooth(read_path='./data/', save_path='./data/', file_name='CNN1.csv')
smooth(read_path='./data/', save_path='./data/', file_name='CNN2.csv')
smooth(read_path='./data/', save_path='./data/', file_name='PPO1.csv')
smooth(read_path='./data/', save_path='./data/', file_name='PPO2.csv')
smooth(read_path='./data/', save_path='./data/', file_name='DQN1.csv')

df1 = pd.read_csv('./data/smooth_Transformer1.csv')
df2 = pd.read_csv('./data/smooth_Transformer2.csv')
df3 = pd.concat([df2, df1]).reset_index(drop=True)
df3.index = range(len(df3))
df4 = pd.read_csv('./data/smooth_CNN1.csv')
df5 = pd.read_csv('./data/smooth_CNN2.csv')
df9 = pd.concat([df4, df5]).reset_index(drop=True)
df6 = pd.read_csv('./data/smooth_PPO1.csv')
df7 = pd.read_csv('./data/smooth_PPO2.csv')
df8 = pd.concat([df6, df7]).reset_index(drop=True)
df10 = pd.read_csv('./data/smooth_DQN1.csv')

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
plt.ylim(0, 4000)
# plt.ylim(0, 500)

# sns.lineplot(x="Number of Iterations", y="Lost Data", data=df17, label='Random', color='violet')
# sns.lineplot(x="Number of Iterations", y="Lost Data", data=df11, label='MA-DQN', color='chocolate')
# sns.lineplot(x="Number of Iterations", y="Lost Data", data=df4, label='SPMA-PPO', color='b')
# sns.lineplot(x="Number of Iterations", y="Lost Data", data=df14, label='CNN-SPMA-PPO', color='g')
# sns.lineplot(x="Number of Iterations", y="Lost Data", data=df8, label='T-SPMA-PPO', color='darkred')

sns.lineplot(x="number of iteration", y="lost data all", data=df3, label='T-SPMA-PPO', color='violet')
sns.lineplot(x="number of iteration", y="lost data all", data=df9, label='CNN-SPMA-PPO', color='b')
sns.lineplot(x="number of iteration", y="lost data all", data=df8, label='SPMA-PPO', color='g')
sns.lineplot(x="number of iteration", y="lost data all", data=df10, label='MA-DQN', color='chocolate')

fig.savefig('reward.pdf', format='pdf', dpi=1200)
plt.show()

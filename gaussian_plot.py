import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns
import pandas as pd


sns.set_theme(style="darkgrid")


df1 = pd.read_csv('./data/PPO_data_0.csv')
df2 = pd.read_csv('./data/PPO_data_1.csv')
# df3 = pd.read_csv('./data/PPO_data_2.csv')
# sns.lineplot(x="number of iteration", y="lost data all", data=data)
df4 = df1.append(df2)
# df5 = df4.append(df3)
df4.index = range(len(df4))
# df4 = df4[df4['Number of Iterations'] > 1000]

df6 = pd.read_csv('./data/lost_data_cpo.csv')
df7 = pd.read_csv('./data/lost_data_cpo2.csv')
df8 = df6.append(df7)
df8.index = range(len(df8))
# df8 = df8[df8['Number of Iterations'] > 1000]

df9 = pd.read_csv('./data/lost_data_dqn1.csv')
df10 = pd.read_csv('./data/lost_data_dqn2.csv')
df11 = df9.append(df10)
df11.index = range(len(df11))
# df11 = df11[df11['Number of Iterations'] > 1000]

df12 = pd.read_csv('./data/CNN_data1.csv')
df13 = pd.read_csv('./data/CNN_data2.csv')
df14 = df12.append(df13)
df14.index = range(len(df14))
# df14 = df14[df14['Number of Iterations'] > 1000]

df15 = pd.read_csv("./data/lost_data_random1.csv")
df16 = pd.read_csv("./data/lost_data_random2.csv")
df17 = df15.append(df16)
df17.index = range(len(df17))


# Transformer数据
data_transformer = df8["Lost Data"]

# PPO数据
data_ppo = df4["Lost Data"]

# DQN数据
data_dqn = df11["Lost Data"]

# CNN数据
data_cnn = df14["Lost Data"]

# Random数据
data_random = df17["Lost Data"]
print(data_random)

# 计算均值和标准差
mean_transformer = np.mean(data_transformer)
std_dev_transformer = np.std(data_transformer)

mean_ppo = np.mean(data_ppo)
std_dev_ppo = np.std(data_ppo)

mean_dqn = np.mean(data_dqn)
std_dev_dqn = np.std(data_dqn)

mean_cnn = np.mean(data_cnn)
std_dev_cnn = np.std(data_cnn)

mean_random = np.mean(data_random)
std_dev_random = np.std(data_random)

# 生成x值
x_transformer = np.linspace(mean_transformer - 3*std_dev_transformer, mean_transformer + 3*std_dev_transformer, 100)
x_ppo = np.linspace(mean_ppo - 3*std_dev_ppo, mean_ppo + 3*std_dev_ppo, 100)
x_dqn = np.linspace(mean_dqn - 3*std_dev_dqn, mean_dqn + 3*std_dev_dqn, 100)
x_cnn = np.linspace(mean_cnn - 3*std_dev_cnn, mean_cnn + 3*std_dev_cnn, 100)
x_random = np.linspace(mean_random - 3*std_dev_random, mean_random + 3*std_dev_random, 100)

# 生成正态分布的y值
y_transformer = norm.pdf(x_transformer, mean_transformer, std_dev_transformer)
y_ppo = norm.pdf(x_ppo, mean_ppo, std_dev_ppo)
y_dqn = norm.pdf(x_dqn, mean_dqn, std_dev_dqn)
y_cnn = norm.pdf(x_cnn, mean_cnn, std_dev_cnn)
y_random = norm.pdf(x_random, mean_random, std_dev_random)

# 绘制图形
plt.plot(x_transformer, y_transformer, label='T-SPMA-PPO', color='darkred')
plt.plot(x_ppo, y_ppo, label='SPMA-PPO', color='b')
plt.plot(x_dqn, y_dqn, label='MA-DQN', color='chocolate')
plt.plot(x_cnn, y_cnn, label='CNN-SPMA-PPO', color='g')
# plt.plot(x_random, y_random, label='Random', color='violet')
# 添加垂直线
plt.axvline(mean_transformer, color='darkred', linestyle='--')
plt.axvline(mean_ppo, color='b', linestyle='--')
plt.axvline(mean_dqn, color='green', linestyle='--')
plt.axvline(mean_cnn, color='g', linestyle='--')

plt.legend()

# 添加x轴和y轴的图例
plt.xlabel('Lost Data')
plt.ylabel('Probability Density')
plt.savefig('g_lost.pdf', format='pdf', dpi=1200)

plt.show()
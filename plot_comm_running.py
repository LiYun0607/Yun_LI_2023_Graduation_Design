import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(style="darkgrid")

# Load the data
df_transformer = pd.read_csv("./data/Transformer_running_data_0.csv")
df_cnn = pd.read_csv("./data/CNN_running_data_1.csv")
df_ppo = pd.read_csv("./data/ppo_running_data_0.csv")
df_random = pd.read_csv("./data/random_running_data_0.csv")

# Define the length of an episode and the episode number for each algorithm
episode_length = 1024
episodes = [0, 3, 10, 1]  # the episode number for each algorithm

# Create a new column for the episode number
df_transformer['episode'] = df_transformer.index // episode_length
df_cnn['episode'] = df_cnn.index // episode_length
df_ppo['episode'] = df_ppo.index // episode_length
df_random['episode'] = df_random.index // episode_length

# Filter data for a specific episode for each algorithm
df_transformer = df_transformer[df_transformer['episode'] == episodes[0]]
df_cnn = df_cnn[df_cnn['episode'] == episodes[1]]
df_ppo = df_ppo[df_ppo['episode'] == episodes[2]]
df_random = df_random[df_random['episode'] == episodes[3]]

# Define the metrics to be plotted
metrics = ["PER", "throughput", "spectral_efficiency", "energy_efficiency", "fairness_index"]
metric_labels = ["Packet Error Rate (PER)", "Throughput (bits/sec)", "Spectral Efficiency (bits/sec/Hz)", "Energy Efficiency (bits/Joule)", "Jain's Fairness Index"]

# Define the dataframes and labels for each algorithm
df_list = [df_transformer, df_cnn, df_ppo, df_random]
labels = ['T-SPMA-PPO', 'CNN-SPMA-PPO', 'SPMA-PPO', 'Random']

# Define a color map for the algorithms
color_map = {'T-SPMA-PPO': 'darkred', 'CNN-SPMA-PPO': 'g', 'SPMA-PPO': 'b', 'Random': 'violet'}

for metric, metric_label in zip(metrics, metric_labels):
    # Create a new figure for each metric
    plt.figure(figsize=(10, 8))

    for df, label in zip(df_list, labels):
        # Draw a line plot for the algorithm
        sns.lineplot(data=df, x=df.index % episode_length, y=metric, alpha=0.3, color=color_map[label])
        # Set y-axis labels with units
        if metric == "PER":
            plt.ylabel("Packet Error Rate (PER)")
        elif metric == "throughput":
            plt.ylabel("Throughput (bits/sec)")
        elif metric == "spectral_efficiency":
            plt.ylabel("Spectral Efficiency (bits/sec/Hz)")
        elif metric == "energy_efficiency":
            plt.ylabel("Energy Efficiency (bits/Joule)")
        elif metric == "fairness_index":
            plt.ylabel("Jain's Fairness Index")

        # Draw a line representing the average value for the algorithm
        plt.plot([0, episode_length], [np.mean(df[metric]), np.mean(df[metric])], label=f'{label} Avg', color=color_map[label], linewidth=2)

    plt.title(f"{metric_label} Over Time Steps")
    plt.legend()
    metric_label = metric_label.replace("/", "_")
    plt.savefig(f'{metric_label}.pdf', format='pdf', dpi=1200)

    plt.show()



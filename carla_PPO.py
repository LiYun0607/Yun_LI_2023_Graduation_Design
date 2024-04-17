import torch.nn as nn
from com_carla_env2 import CarlaEnv
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Create environment
env = CarlaEnv()


def learning_rate_schedule(progress_remaining):
    return 0.0003 * progress_remaining

reward_ = env.reward_
lost_data_all = env.lost_data_all
# Train agent
model = PPO("MlpPolicy", env, verbose=1, learning_rate=learning_rate_schedule)
model.learn(total_timesteps=3072000)
model.save("MAPPO")

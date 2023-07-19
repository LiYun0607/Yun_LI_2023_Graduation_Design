import torch.nn as nn
from com_carla_env_mul import CarlaEnv
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np


def learning_rate_schedule(progress_remaining):
    return 0.0003 * progress_remaining


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.rewards = []
        self.lost_data = []

    def _on_step(self) -> bool:
        # Check if an episode has ended
        done = self.locals.get("done")
        if done is not None and done:
            # Get the info dictionaries for each environment
            infos = self.locals.get("infos")

            # Get the 'reward' and 'data_loss' values from the info dictionary of each environment
            rewards = [info.get('reward') for info in infos]
            lost_data = [info.get('data_loss') for info in infos]

            # Calculate and record the average reward and data loss
            self.rewards.append(np.mean(rewards))
            self.lost_data.append(np.mean(lost_data))
            print(f"rewards: {self.rewards[-1]}, lost_data: {self.lost_data[-1]}")
        return True


class CustomTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomTransformer, self).__init__(observation_space, features_dim)

        # Define the transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=7, nhead=7)  # Set d_model equal to the feature size
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

        # Define the output layer
        self.linear = nn.Linear(7 * 30, features_dim)  # Change the input size of the linear layer

    def forward(self, observations):
        observations = observations.view(observations.shape[0], 30, -1)  # Reshape to (batch_size, n_vehicles, 2 + queue_length)
        transformer_output = self.transformer_encoder(observations)
        return self.linear(transformer_output.view(transformer_output.shape[0], -1))

def make_env(port):
    def _init():
        return CarlaEnv(port=port)
    return _init


if __name__ == '__main__':
    ports = [2000, 2004]
    envs = SubprocVecEnv([make_env(port) for port in ports])
    total_rewards = np.zeros(2)  # 用于保存每个环境的总奖励
    total_lost_data = np.zeros(2)  # 用于保存每个环境的总丢失数据
    # Define policy with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomTransformer,
        features_extractor_kwargs=dict(features_dim=64),
    )
    # reward_ = env.reward_
    # lost_data_all = env.lost_data_all
    # Train agent
    # model = PPO("MlpPolicy", envs, verbose=1, policy_kwargs=policy_kwargs, n_steps=1024, learning_rate=learning_rate_schedule)
    # model.learn(total_timesteps=3072000)
    callback = CustomCallback()
    model = PPO("MlpPolicy", envs, verbose=1, policy_kwargs=policy_kwargs, n_steps=1024,
                learning_rate=learning_rate_schedule)
    model.learn(total_timesteps=3072000, callback=callback)

    # Now you can access the recorded rewards and lost data
    print(callback.rewards)
    print(callback.lost_data)

# # Create environment
# env = CarlaEnv()
#
# # Define policy with custom feature extractor
# policy_kwargs = dict(
#     features_extractor_class=CustomTransformer,
#     features_extractor_kwargs=dict(features_dim=64),
# )
# reward_ = env.reward_
# lost_data_all = env.lost_data_all
# # Train agent
# model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
# model.learn(total_timesteps=3072000)

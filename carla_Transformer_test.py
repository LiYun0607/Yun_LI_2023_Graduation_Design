import torch.nn as nn
from com_carla_env_test import CarlaEnv
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


            # Calculate and record the average reward and data loss

        return True


class CustomTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomTransformer, self).__init__(observation_space, features_dim)

        # Define the transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=3, nhead=3)  # Set d_model equal to the feature size
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

        # Define the output layer
        self.linear = nn.Linear(7 * 30, features_dim)  # Change the input size of the linear layer

    def forward(self, observations):
        observations = observations.view(observations.shape[0], 70, -1)  # Reshape to (batch_size, n_vehicles, 2 + queue_length)
        transformer_output = self.transformer_encoder(observations)
        return self.linear(transformer_output.view(transformer_output.shape[0], -1))

def make_env(port, env_id, tm_port):
    def _init():
        return CarlaEnv(port=port, env_id=env_id, tm_port=tm_port)
    return _init


if __name__ == '__main__':
    ports = [2000, 2004]
    tm_ports = [8000, 8004]
    envs = SubprocVecEnv([make_env(port, i, tm_port) for i, (port, tm_port) in enumerate(zip(ports, tm_ports))])
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
    model = PPO.load("Transformer_PPO", env=envs, verbose=1, n_steps=2048, learning_rate=learning_rate_schedule)
    model.learn(total_timesteps=20480, callback=callback)

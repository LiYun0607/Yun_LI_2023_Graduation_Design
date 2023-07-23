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


class CustomConv1D(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomConv1D, self).__init__(observation_space, features_dim)

        # Define the Conv1D architecture
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3264, features_dim)  # Change the input size according to the output of the conv layers
        )

    def forward(self, observations):
        # observations shape needs to be (batch_size, channels, length)
        observations = observations.unsqueeze(1)  # Adds the channel dimension
        return self.conv1d(observations)


def make_env(port, env_id):
    def _init():
        return CarlaEnv(port=port, env_id=env_id)
    return _init


if __name__ == '__main__':
    ports = [2000, 2004]
    envs = SubprocVecEnv([make_env(port, i) for i, port in enumerate(ports)])
    total_rewards = np.zeros(2)  # 用于保存每个环境的总奖励
    total_lost_data = np.zeros(2)  # 用于保存每个环境的总丢失数据
    # Define policy with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomConv1D,
        features_extractor_kwargs=dict(features_dim=64),
    )
    model = PPO("MlpPolicy", envs, verbose=1, policy_kwargs=policy_kwargs, n_steps=1024)

    model.learn(total_timesteps=3072000)
    model.save("ppo_CNN")


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

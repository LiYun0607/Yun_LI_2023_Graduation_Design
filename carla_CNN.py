import torch.nn as nn
from com_carla_env2 import CarlaEnv
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomConv1D(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
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


# Create environment
env = CarlaEnv()


def learning_rate_schedule(progress_remaining):
    return 0.0003 * progress_remaining

# Define policy with custom feature extractor
policy_kwargs = dict(
    features_extractor_class=CustomConv1D,
    features_extractor_kwargs=dict(features_dim=64),
    # optimizer_kwargs=dict(weight_decay=0.00001),
)
reward_ = env.reward_
lost_data_all = env.lost_data_all
# Train agent
model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
            learning_rate=learning_rate_schedule)

model.learn(total_timesteps=3072000)
model.save("ppo_CNN")

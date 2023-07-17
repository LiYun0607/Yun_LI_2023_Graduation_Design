import torch.nn as nn
from com_carla_env import CarlaEnv
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomTransformer, self).__init__(observation_space, features_dim)

        # Define the transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=7, nhead=7)  # Set d_model equal to the feature size
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

        # Define the output layer
        self.linear = nn.Linear(7 * env.n_vehicles, features_dim)  # Change the input size of the linear layer

    def forward(self, observations):
        observations = observations.view(observations.shape[0], env.n_vehicles, -1)  # Reshape to (batch_size, n_vehicles, 2 + queue_length)
        transformer_output = self.transformer_encoder(observations)
        return self.linear(transformer_output.view(transformer_output.shape[0], -1))


# Create environment
env = CarlaEnv()

# Define policy with custom feature extractor
policy_kwargs = dict(
    features_extractor_class=CustomTransformer,
    features_extractor_kwargs=dict(features_dim=64),
)
reward_ = env.reward_
lost_data_all = env.lost_data_all
# Train agent
model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=3072000)

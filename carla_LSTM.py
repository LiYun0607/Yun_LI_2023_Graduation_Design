import torch.nn as nn
from com_carla_env2 import CarlaEnv
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomLSTM(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomLSTM, self).__init__(observation_space, features_dim)
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=7, hidden_size=64, num_layers=2, batch_first=True)
        # Define the output layer
        self.linear = nn.Linear(64, features_dim)

    def forward(self, observations):
        observations = observations.view(observations.shape[0], env.n_vehicles, -1) # Reshape to (batch_size, n_vehicles, 2 + queue_length)
        lstm_output, _ = self.lstm(observations)
        return self.linear(lstm_output[:, -1, :]) # Use the last hidden state for output

# Create environment
env = CarlaEnv()

def learning_rate_schedule(progress_remaining):
    return 0.0003 * progress_remaining

# Define policy with custom feature extractor
policy_kwargs = dict(
    features_extractor_class=CustomLSTM,
    features_extractor_kwargs=dict(features_dim=64),
    # optimizer_kwargs=dict(weight_decay=0.00001),
)

reward_ = env.reward_
lost_data_all = env.lost_data_all

# Train agent
model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, learning_rate=learning_rate_schedule)
model.learn(total_timesteps=3072000)

from torch.nn import functional
from torch import jit, nn
import torch
import math

LOG_STD_MAX = 0
LOG_STD_MIN = -1
EPS = 1e-8
log_std_init = 0
running_time = 72
step_time = 0.1

def initWeights(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(0, 0.01)

def initWeights2(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(-1.0, 0.01)

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()

        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.hidden1_units = args['hidden1']
        self.hidden2_units = args['hidden2']

        self.fc1 = nn.Linear(self.state_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.act_fn = torch.relu
        self.output_act_fn = torch.sigmoid

        self.fc_mean = nn.Linear(self.hidden2_units, self.action_dim)
        # self.fc_log_std = nn.Linear(self.hidden2_units, self.action_dim)
        self.count = 0

    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        mean = self.output_act_fn(self.fc_mean(x))
        # log_std = self.fc_log_std(x)
        # print("log_std", log_std)

        # log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        # print("log_std_revised", log_std)
        # std = torch.exp(log_std)
        # log_std = nn.Parameter(torch.ones(self.action_dim)*log_std_init)
        total_steps = running_time*3600/step_time
        log_std_init = math.exp(-1.5e-6*self.count)
        std = nn.Parameter(torch.ones(self.action_dim)*log_std_init)
        # std = torch.exp(log_std)
        # if epoch > 300:
        #     log_std_ = nn.Parameter(torch.ones(self.action_dim)*-4.1864)
        #     std = torch.exp(log_std_)
        # else:
        #     std = std_init - self.count / total_steps
        self.count += 1
        # print("std", std, std.type())

        return mean, std

    def initialize(self):
        for m_idx, module in enumerate(self.children()):
            if m_idx != 3:
                module.apply(initWeights)
            else:
                module.apply(initWeights2)


class Value(nn.Module):
    def __init__(self, args):
        super(Value, self).__init__()

        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.hidden1_units = args['hidden1']
        self.hidden2_units = args['hidden2']

        self.fc1 = nn.Linear(self.state_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.fc3 = nn.Linear(self.hidden2_units, 1)
        self.act_fn = torch.relu


    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.fc3(x)
        x = torch.reshape(x, (-1,))
        return x

    def initialize(self):
        self.apply(initWeights)

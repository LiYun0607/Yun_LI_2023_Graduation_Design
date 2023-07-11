from cpo_pre import Agent
from com_carla_env import CarlaEnv
import numpy as np
import argparse
import pickle
import random
import torch
import wandb


wandb.init(project="v2x_cr_cpo")


def train(main_args):
    algo_idx = 1
    agent_name = 'CPO'
    env_name = "CarlaCPO"
    max_ep_len = 2048
    max_steps = 2048
    epochs = 1500
    save_freq = 1
    algo = '{}_{}'.format(agent_name, algo_idx)
    save_name = '_'.join(env_name.split('-')[:-1])
    save_name = "result/{}_{}".format(save_name, algo)
    args = {
        'agent_name':agent_name,
        'save_name': save_name,
        'discount_factor':0.99,
        'hidden1':128,
        'hidden2':128,
        'v_lr':3e-4,
        'cost_v_lr':3e-4,
        'value_epochs':1500,
        'batch_size':2048,
        'num_conjugate':10,
        'max_decay_num':10,
        'line_decay':0.8,
        'max_kl':0.001,
        'damping_coeff':0.01,
        'gae_coeff':0.97,
        'cost_d':0.01,
    }
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')

    # for random seed
    seed = algo_idx + random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)

    env = CarlaEnv()
    agent = Agent(env, device, args)

    for epoch in range(epochs):
        trajectories = []
        ep_step = 0
        collision_n = 0
        step_ = []
        d_acceleration_ = []
        episode_length = 0
        while ep_step < max_steps:
            state = env.reset()
            score = 0
            cv = 0
            step = 0
            cost_ = 0
            d_acceleration = 0
            while True:
                step += 1
                ep_step += 1
                state_tensor = torch.tensor(state, device=device, dtype=torch.float32)
                action_tensor, clipped_action_tensor = agent.getAction(state_tensor, is_train=True)
                action = action_tensor.detach().cpu().numpy()
                # print(action)
                clipped_action = clipped_action_tensor.detach().cpu().numpy()
                next_state, reward, done, info = env.step(clipped_action)

                cost = info['cost']
                cost_ += cost

                done = True if step >= max_ep_len else done
                fail = True if step < max_ep_len and done else False
                trajectories.append([state, action, reward, cost, done, fail, next_state])

                state = next_state
                score += reward
                score = round(score, 3)

                if done or step >= max_ep_len:
                    episode_length += 1
                    step_.append(step)
                    d_acceleration_.append(d_acceleration)
                    break
            v_loss, cost_v_loss, objective, cost_surrogate, kl, entropy, grad_g, grad_b, optimization_case = agent.train(
                trajs=trajectories)
            reward_ = env.reward_
            lost_data_all = env.lost_data_all
            wandb.log({
                "v_loss": v_loss,
                "cost_v_loss": cost_v_loss,
                "objective": objective,
                "cost_surrogate": cost_surrogate,
                "kl": kl,
                "entropy": entropy,
                "grad_g": grad_g,
                "grad_b": grad_b,
                "optimization_case": optimization_case,
                "reward": reward_[-1] if reward_ else None,  # 记录最新的reward值
                "lost_data_all": lost_data_all[-1] if lost_data_all else None  # 记录最新的lost_data_all值
            })


def test(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPO')
    parser.add_argument('--test', action='store_true', help='For test.')
    parser.add_argument('--resume', type=int, default=0, help='type # of checkpoint.')
    parser.add_argument('--graph', action='store_true', help='For graph.')
    args = parser.parse_args()
    dict_args = vars(args)
    if args.test:
        test(args)
    else:
        train(args)

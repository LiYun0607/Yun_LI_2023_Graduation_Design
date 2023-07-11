import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import spaces
import abc
import glob
import os
import sys
import argparse
import pandas as pd

from types import LambdaType
from collections import deque
from collections import namedtuple

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])

except IndexError:
    pass

import carla
import random
import time
import math
import torch
from carla import Transform, Location, Rotation
import wandb
from agents.navigation.basic_agent import BasicAgent


wandb.init(project="cr-cpo")
argparser = argparse.ArgumentParser(description="args for carla environment")
argparser.add_argument(
    '--host',
    metavar='H',
    default='127.0.0.1',
    help='IP of the host server (default: 127.0.0.1)')
argparser.add_argument(
    '-p', '--port',
    metavar='P',
    default=2000,
    type=int,
    help='TCP port to listen to (default: 2000)')
argparser.add_argument(
    '--filterv',
    metavar='PATTERN',
    default='vehicle.*',
    help='Filter vehicle model (default: "vehicle.*")')
argparser.add_argument(
    '--generationv',
    metavar='G',
    default='All',
    help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
argparser.add_argument(
    '-n', '--number-of-vehicles',
    metavar='N',
    default=30,
    type=int,
    help='Number of vehicles (default: 30)')
argparser.add_argument(
    '-s', '--number-of-episodes',
    metavar='S',
    default=2048,
    type=int,
    help='Number of steps per epoch')

args = argparser.parse_args()

TOWN = 'Town02'
tau = 0.1
n_lic_channels = 4
n_decimals = 1
log_dir = './data/'
os.makedirs(log_dir, exist_ok=True)
# locations of vehicles: x, y, rotation
init_loc = np.array([[245, -246, 0],
                     [225, -246, 0],
                     [255.5, -291, 90],
                     [255.5, -304, 90],
                     [295, -250, 180],
                     [319, -250, 180],
                     [258.5, -185, 270],
                     [258.5, -201, 270]])
tar_loc = np.array([[255, -155, 90],
                    [238.5, -310, 180],
                    [245.5, -175, 90],
                    [205, -260, 180],
                    [200, -220, 180],
                    [278.5, -306, 0],
                    [300, -246, 0],
                    [185, -252, 180]])

# units of parameters: bandwidth: MHz; path loss per meter: dB; power: dBm
class DataRate():
    def __init__(self):
        self.bandwidth_v2i = 1e7  # MHz
        self.bandwidth_cr = 1e7
        L0 = 47.86  # dB
        self.L0 = 10**(L0/10)
        self.alpha = 2.75
        self.v2i_power = 33  # dBm
        self.v2v_power = 30
        self.awgn_power = -95  # dBm
        self.lambda_on = 0.5
        self.lambda_off = 0.5
        self.v2v_range = 8

    # calculate integration
    def cal_integral(self, begin, end, epsilon, v2i_rate):
        def func(x):
            return x * self.bandwidth_cr * self.lambda_on * v2i_rate * math.exp(
                -self.lambda_off * x) / (
                    tau * self.bandwidth_v2i * (self.lambda_on + self.lambda_off))

        def simpson(begin, between, i):
            a = begin + (i - 1) * between
            b = begin + i * between
            return between * (func(a) + func(b) + 4 * func((a + b) / 2)) / 6

        n = 1
        result = 0
        preResult = float("inf")
        while abs(preResult - result) >= epsilon:
            preResult = result
            result = 0
            n *= 2
            between = (end - begin) / n
            for i in range(n):
                try:
                    result += simpson(begin, between, i + 1)
                except:
                    return "Integrated function has discontinuity or does not " \
                           "defined in current interval"
        return result

    def v2i_rate(self, d):
        v2i_rate = self.bandwidth_v2i * math.log(
            1 + 10 ** ((self.v2i_power - self.awgn_power) / 10 - 3) /
            (self.L0 * d ** self.alpha), 2)
        return v2i_rate

    def cr_rate(self, v2i_rate):
        cr_rate = self.cal_integral(0, tau, 0.001, v2i_rate)
        return cr_rate

    def v2v_rate(self, j, loc_all, actions):
        v_chosen = None
        a = 0
        v2v_rate = 0
        v2v_candidates = {
            "dist": [],
            "id": [],
        }
        for k in range(args.number_of_vehicles):
            # find vehicles set within the V2V communication range and
            # don't take any actions
            if np.linalg.norm(loc_all[j] - loc_all[k]) <= self.v2v_range and k != j \
                    and actions[k] == 3:
                v2v_candidates['dist'].append(np.linalg.norm(loc_all[j] - loc_all[k]))
                v2v_candidates['id'].append(k)
        # find which v' to communicate for v
        if len(v2v_candidates['dist']) > 0:
            d = np.min(v2v_candidates['dist'])
            id_chosen = np.where(v2v_candidates['dist'] == d)[0][0]
            v_chosen = v2v_candidates['id'][id_chosen]
            # remove d and id from candidate vehicle set
            v2v_candidates['dist'].remove(d)
            v2v_candidates['id'].remove(v_chosen)
            # interference_dist.remove(d)
            if len(v2v_candidates['id']) > 0:
                for i in range(len(v2v_candidates['id'])):
                    d_ = np.linalg.norm(loc_all[v_chosen] -
                                        loc_all[v2v_candidates['id'][i]])
                    a += 10 ** ((self.v2v_power - self.awgn_power) / 10 - 3) / \
                         (self.L0 * d_ ** self.alpha)
                v2v_rate = self.bandwidth_v2i * math.log(
                    1 + ((10 ** ((self.v2v_power - self.awgn_power) / 10 - 3) /
                          (self.L0 * d ** self.alpha)) / (1 + a)))
        return v2v_rate, v_chosen


class GenerateData:
    def __init__(self):
        self.n_data_type = 3
        self.data_len_max = 30
        self.data_len_min = 10
        self.data_cons_min = 1
        self.data_cons_max = 5

    def generate_data(self):
        data_produce_probability = np.array(
            np.random.random(self.n_data_type))
        data_type = np.where(
            data_produce_probability == np.max(data_produce_probability))
        d_amount = round(
            np.array(np.random.randint(self.data_len_min, self.data_len_max)) *
            np.max(data_produce_probability), n_decimals)
        d_cons = np.array(
            [np.random.randint(self.data_cons_min, self.data_cons_max + 1)])

        data = {
            "data amount": d_amount,
            "data constraint": d_cons,
            # type is related with cons
            "data type": data_type,
        }
        return data


# build cache queue state for one vehicle
class CacheQueue:
    def __init__(self):
        self.queue_len_max = 5
        self.queue_state = np.zeros(self.queue_len_max)

    def queue_in(self, queue, data_amount, data_cons):
        self.queue_state = np.zeros(self.queue_len_max)
        for i in range(len(queue) - 1):
            self.queue_state[i] = queue[i + 1]
        self.queue_state[data_cons - 1] += data_amount
        return self.queue_state


class CarlaEnv(gym.Env):

    def __init__(self):
        super(CarlaEnv, self).__init__()
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(300.0)
        self.world = self.client.load_world(TOWN)
        settings = self.world.get_settings()
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            # 20fps
            settings.fixed_delta_seconds = 0.1
            self.world.apply_settings(settings)
        self.synchronous_master = True
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(settings)

        # --------------
        # Spawn vehicles
        # --------------
        self.batch = []
        self.vehicles_list = []
        self.spawn_random_traffic()

        for response in self.client.apply_batch_sync(self.batch, self.synchronous_master):
            self.vehicles_list.append(response.actor_id)

        self.n_vehicles = len(self.batch)
        self.loc_min = -1000
        self.loc_max = 1000
        self.n_actions = 4

        self.data_generator = GenerateData()
        self.cache_generator = CacheQueue()
        low_observation = []
        high_observation = []
        low_action = []
        high_action = []
        action_space = []
        self.untransmiteed_data = []
        self.lost_data_all = []
        self.cache_queue = []
        self.data_all = []
        self.location_all = np.zeros([self.n_vehicles, 2])
        self.cal_data_rate = DataRate()
        self.n_v2i_channel = self.cal_data_rate.lambda_off ** -1 * \
                             n_lic_channels
        self.n_v2v_channel = self.cal_data_rate.lambda_on ** -1 * \
                             n_lic_channels
        self.n_cr_channel = 2
        self.cost_lic = -1.5
        self.cost_cr = -0.5
        self.penalty = -0.4
        self.location_rsu = (0, 200)
        # append queue state for each vehicle
        for i in range(self.n_vehicles * self.data_generator.data_cons_max):
            low_observation.append(0)
            high_observation.append(self.data_generator.data_len_max)
        # append location state (x,y) for each vehicle
        for i in range(self.n_vehicles * 2):
            low_observation.append(self.loc_min)
            high_observation.append(self.loc_max)
        for i in range(self.n_vehicles):
            low_action.append(0)
            high_action.append(4)

        self.observation_space = spaces.Box(
            low=np.array(low_observation, dtype=np.float32),
            high=np.array(high_observation, dtype=np.float32))
        self.action_space = spaces.Box(
            low=np.array(low_action, dtype=np.float32),
            high=np.array(high_action, dtype=np.float32))
        self.state = None
        self.xs = np.zeros(self.n_vehicles)
        self.ys = np.zeros(self.n_vehicles)
        self.collision_hist = []
        self.collision_transform = []
        self.collision_sensors = []
        self.collision_hist_l = 1
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.time_step = 0
        self.n_itr = 0
        self.n_itr_ = []
        self.reward = []
        self.reward_ = []
        self.x_min = []
        self.x_max = []
        self.y_min = []
        self.y_max = []

        def get_collision_hist(event):
            impulse = event.normal_impulse
            # print(event.transform.location, event.actor.type_id)
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)
        for actor in self.world.get_actors():
            if len(actor.semantic_tags) > 0:
                if actor.semantic_tags[0] == 10:
                    collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(),
                                                                    attach_to=actor)
                    self.collision_sensors.append(collision_sensor)
                    self.collision_sensors[-1].listen(lambda event: get_collision_hist(event))

    def spawn_random_traffic(self):
        self.batch = []
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(3)
        traffic_manager.global_percentage_speed_difference(30.0)
        def get_actor_blueprints(world, filter, generation):
            bps = world.get_blueprint_library().filter(filter)
            if generation.lower() == "all":
                return bps
            if len(bps) == 1:
                return bps
            try:
                int_generation = int(generation)
                # Check if generation is in available generations
                if int_generation in [1, 2]:
                    bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
                    return bps
                else:
                    print("   Warning! Actor Generation is not valid. No actor will be spawned.")
                    return []
            except:
                print("   Warning! Actor Generation is not valid. No actor will be spawned.")
                return []

        blueprints = get_actor_blueprints(self.world, args.filterv, args.generationv)
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
        blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]
        self.blueprints = sorted(blueprints, key=lambda bp: bp.id)
        self.spawn_points = self.world.get_map().get_spawn_points()
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        for n, transform in enumerate(self.spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(self.blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            self.batch.append(SpawnActor(blueprint, transform)
                              .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    def reset(self):
        self.time_step = 0
        self.untransmiteed_data = []
        self.reward = []
        # DestroyActor = carla.command.DestroyActor
        # for actor in self.world.get_actors():
        #     DestroyActor(actor)
        # print("destroy actors!")
        # self.spawn_random_traffic()
        # print("spawn actors")
        self.cache_queue = np.zeros(
            [self.n_vehicles, self.data_generator.data_cons_max])
        self.location_all = np.zeros([self.n_vehicles, 2])
        self.get_location()
        self.state = []
        for i in range(self.n_vehicles):
            self.state.append(self.location_all[i][0])
            self.state.append(self.location_all[i][1])
            for j in range(self.data_generator.data_cons_max):
                self.state.append(self.cache_queue[i][j])
        return np.array(self.state, dtype=np.float32)

    def state_transition(self, cache_queues, actions, data):
        data_loss_total = 0
        cost_total = 0
        new_queues = np.array(list(cache_queues))
        max_length = self.cache_generator.queue_len_max
        # number of data transmitted from vehicle i to RSU m through V2I (licensed and CR)
        d_v2i = np.zeros(self.n_vehicles)
        d_v2v_out = np.zeros(self.n_vehicles)
        d_v2v_in = np.zeros(self.n_vehicles)
        d_lost = np.zeros(self.n_vehicles)

        n_vehicle_lic_v2i = len(list(np.where(actions == 0))[0])
        n_vehicle_cr_v2i = len(list(np.where(actions == 1))[0])

        for i in range(self.n_vehicles):
            # self.x_min.append(self.location_all[i][0])
            # self.y_min.append(self.location_all[i][1])
            # print(f"x min: {min(self.x_min)}, x_max: {max(self.x_min)}, y min: {min(self.y_min)},
            # y_max: {max(self.y_min)}")
            # if i == 0:
            #     print(f"not updated cache state: {cache_queues[0]}")
            # the minimum non-empty index
            # if the queue is empty, it can only receive data from v2v or generated by itself
            if len(np.where(cache_queues[i] == 0)[0]) == self.cache_generator.queue_len_max:
                new_queues[i][1] += d_v2v_in[i]
                new_queues[i][data[i]['data constraint'] - 1] += data[i]['data amount']
                continue
            t_min = min(np.where(cache_queues[i] != 0)[0])
            # settings to make the experiment easier
            d = np.linalg.norm(self.location_all[i] - self.location_rsu[0]) * 10  # one cell is 10 meters
            if n_vehicle_lic_v2i > 0 and n_vehicle_cr_v2i > 0:
                v2i_licensed = self.cal_data_rate.v2i_rate(d)
                d_v2i_ = round(v2i_licensed * tau * self.n_v2i_channel / n_vehicle_lic_v2i, 1)
                v2i_cr = self.cal_data_rate.cr_rate(d)
                d_cr_ = round(v2i_cr * tau * self.n_cr_channel / n_vehicle_cr_v2i, 1)
                if d_v2i_ + (self.cost_lic-self.cost_cr)/self.penalty < d_cr_ < cache_queues[i][0]:
                    actions[i] = 0
                if d_v2i_ < cache_queues[i][0] - (self.cost_lic-self.cost_cr)/self.penalty and d_cr_ > cache_queues[i][0]:
                    actions[i] = 1
            # vehicle i chose licensed V2I communication
            if actions[i] == 0:
                d = np.linalg.norm(self.location_all[i] - self.location_rsu[0]) * 10  # one cell is 10 meters
                v2i_licensed = self.cal_data_rate.v2i_rate(d)
                d_v2i[i] += round(v2i_licensed * tau * self.n_v2i_channel / n_vehicle_lic_v2i, 1)
                d_lost[i] = max(0, cache_queues[i][t_min] - d_v2i[i])
                cache_queues[i][t_min] = 0
                cost_total += self.cost_lic
            # vehicle i chose CR communication
            if actions[i] == 1:
                d = np.linalg.norm(self.location_all[i] - self.location_rsu[0]) * 10
                v2i_cr = self.cal_data_rate.cr_rate(d)
                d_v2i[i] += round(v2i_cr * tau * self.n_cr_channel / n_vehicle_cr_v2i, 1)
                d_lost[i] = max(0, cache_queues[i][t_min] - d_v2i[i])
                cache_queues[i][t_min] = 0
                cost_total += self.cost_lic
            # vehicle i chose V2V communication, data send out
            if actions[i] == 2:
                v2v_rate, v_chosen = self.cal_data_rate.v2v_rate(i, self.location_all, actions)
                d_v2v_out[i] = round(v2v_rate * tau / self.n_v2v_channel, 1)
                # vehicle i transmit data #d to vehicle v_chosen, so queue i minus #d, queue v_chosen increase #d
                d_v2v_in[v_chosen] = d_v2v_out[i]
                cost_total += self.cost_cr
                d_lost[i] = cache_queues[i][0]
            if actions[i] == 3:
                d_lost[i] = cache_queues[i][0]

            # state transition: t_min

            if t_min == data[i]['data constraint'] - 1:
                n_data = data[i]['data amount']
            else:
                n_data = 0
            if t_min == 0:
                new_queues[i][t_min] = cache_queues[i][t_min+1] + n_data
            if t_min == max_length - 1:
                new_queues[i][t_min] = max(
                    0, n_data - d_v2i[i] - d_v2v_out[i])
            else:
                new_queues[i][t_min] = max(
                    0, cache_queues[i][t_min+1] + n_data)

            # state transition: remaining t except t_min
            for t in range(max_length):
                if t != t_min:
                    if t == data[i]['data constraint'] - 1:
                        n_data = data[i]['data amount']
                    else:
                        n_data = 0
                    if t == max_length - 1:
                        new_queues[i][t] = n_data
                    else:
                        new_queues[i][t] = n_data + cache_queues[i][t + 1]
            data_loss_total = np.sum(d_lost)
        reward = self.penalty * data_loss_total + cost_total

        # print(f"vehicle id: {0}, action: {actions[0]}\n"
        #       f"updated cache state: {new_queues[0]}, \n"
        #       f"generated data: {round(data[0]['data amount'], 1)}, cons: {data[0]['data constraint']}\n"
        #       f"data from other vehicle: {round(d_v2v_in[0], 1)}, \n"
        #       f"data transmit via V2I: {d_v2i[0]}, \n"
        #       f"data lost: {d_lost[0]}, \n"
        #       f"data transmit via V2V: {d_v2v_out[0]} \n")
        return new_queues, reward, data_loss_total

    def detect_collision(self):
        def get_collision_hist(event):
            impulse = event.normal_impulse
            # print(event.transform.location, event.actor.type_id)
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

    def get_location(self):
        count = 0
        for actor in self.world.get_actors():
            if len(actor.semantic_tags) > 0:
                if actor.semantic_tags[0] == 10:
                    # collision_sensor = (self.world.spawn_actor(self.collision_bp, carla.Transform(),
                    #                                                 attach_to=actor))
                    self.xs[count] = round(actor.get_location().x, 1)
                    self.ys[count] = round(actor.get_location().y, 1)
                    # if len(self.collision_hist) > 0:
                    #     location = random.choice(self.spawn_points).location
                    #     actor.set_location(location)
                    count += 1
        for i in range(self.n_vehicles):
            self.data_all.append(self.data_generator.generate_data())
            self.location_all[i][0] = self.xs[i]
            self.location_all[i][1] = self.ys[i]

    def step(self, action):
        action = action / 4.0 * 3.0  # 确保动作值在 0 到 3 之间
        discrete_action = np.floor(action).astype(int)
        self.world.tick()
        info = {}
        info['cost'] = 0
        self.time_step += 1
        xs = np.zeros(self.n_vehicles)
        ys = np.zeros(self.n_vehicles)
        count = 0
        for actor in self.world.get_actors():
            if len(actor.semantic_tags) > 0:
                if actor.semantic_tags[0] == 10:
                    xs[count] = actor.get_location().x
                    ys[count] = actor.get_location().y
                    count += 1
        done = False
        info = {}
        state = []
        self.data_all = []
        # generate random data for each vehicle
        self.get_location()
        self.detect_collision()
        self.cache_queue, reward, data_loss_total = self.state_transition(self.cache_queue, action, self.data_all)
        info['cost'] = data_loss_total
        self.reward.append(reward)
        self.untransmiteed_data.append(data_loss_total)
        for i in range(self.n_vehicles):
            state.append(self.location_all[i][0])
            state.append(self.location_all[i][1])
            for j in range(self.data_generator.data_cons_max):
                state.append(self.cache_queue[i][j])
        print(f"time step: {self.time_step}, reward: {reward}, cost: {info['cost']}")
        if self.time_step >= args.number_of_episodes:
            self.n_itr += 1
            self.n_itr_.append(self.n_itr)
            done = True
            lost_data_all = np.mean(self.untransmiteed_data)
            self.lost_data_all.append(lost_data_all)
            self.reward_.append(np.mean(self.reward))
            log_data = pd.DataFrame({
                "lost data all": self.lost_data_all,
                "number of iteration": self.n_itr_,
                "reward": self.reward_,
            })
            log_dir_ = os.path.join(log_dir, 'lost_data_cpo.csv')
            log_data.to_csv(log_dir_)
            for i in range(len(self.lost_data_all)):
                wandb.log({
                    "lost data all": self.lost_data_all[i],
                    "number of iteration": self.n_itr_[i],
                    "reward": self.reward_[i],
                })

            print("write file data successfully!")

        # print(f"state: {state}")
        # print(f"cache queue: {self.cache_queue}")
        # print(f"locations: {self.location_all}")
        # print(f"data lost: {data_loss_total}, reward: {reward}")

        return np.array(state, dtype=np.float32), reward, done, info


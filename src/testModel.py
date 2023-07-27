# from madrl_environments.pursuit import MAWaterWorld_mod
import robot_exploration_v1
from maddpg.MADDPG import MADDPG
import numpy as np
import torch as th
import torch.nn as nn
from tensorboardX import SummaryWriter
from copy import copy,deepcopy
from torch.distributions import categorical
from sim_utils import onehot_from_action
import time
import os
import yaml
from datetime import datetime




print(th.cuda.is_available())
# do not render the scene
e_render = True
# tensorboard writer
time_now = time.strftime("%m%d_%H%M%S")

writer = SummaryWriter(os.getcwd()+'/../runs/'+time_now)

num_step_file = open(os.getcwd()+'/../runs/'+time_now+'/num_steps.txt', "w")
num_step_file.close()

food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
n_coop = 2
world = robot_exploration_v1.RobotExplorationT1()
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
world.seed(1234)
n_agents = world.number
n_states = world.number
n_actions = 8
n_pose = 2
# capacity = 1000000
capacity = 5000
# batch_size = 1000
batch_size = 1

n_episode = 1000
# max_steps = 1000
max_steps = 50
# episodes_before_train = 1000
episodes_before_train = 1

model_save_eps = 10


win = None
param = None
avg = None
load_model = True

MODEL_PATH1 = r'E:\Summer Research 2023\MADDPG\MADDPG\model\2023_07_26_21_45_55\model-120.pth'
MODEL_PATH2 = r'E:\Summer Research 2023\MADDPG\MADDPG\model\2023_07_26_21_45_55\model-500.pth'
CONFIG_PATH = os.getcwd() + '/../assets/config.yaml'
current_time = datetime.now()
time_string = current_time.strftime('%Y_%m_%d_%H_%M_%S')
MODEL_DIR = os.getcwd() + '/../model/' + time_string


maddpg1 = MADDPG(n_agents, n_states, n_actions, n_pose, batch_size, capacity,
                episodes_before_train)

maddpg2 = MADDPG(n_agents, n_states, n_actions, n_pose, batch_size, capacity,
                episodes_before_train)


with open(CONFIG_PATH,'r') as stream:
    config = yaml.safe_load(stream)
checkpoints1 = th.load(MODEL_PATH1)
checkpoints2 = th.load(MODEL_PATH2)

actor1 = checkpoints1['actor_%d' % (0)]
actor2 = checkpoints2['actor_%d' % (0)]
print("hello")


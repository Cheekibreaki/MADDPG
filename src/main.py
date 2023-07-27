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
episodes_before_train = 10

model_save_eps = 10


win = None
param = None
avg = None
load_model = True

MODEL_PATH = r'E:\Summer Research 2023\MADDPG\MADDPG\model\2023_07_26_21_45_55\model-100.pth'
CONFIG_PATH = os.getcwd() + '/../assets/config.yaml'
current_time = datetime.now()
time_string = current_time.strftime('%Y_%m_%d_%H_%M_%S')
MODEL_DIR = os.getcwd() + '/../model/' + time_string

maddpg = MADDPG(n_agents, n_states, n_actions, n_pose, batch_size, capacity,
                episodes_before_train)
with open(CONFIG_PATH,'r') as stream:
    config = yaml.safe_load(stream)

if load_model:

    print("loaded")
    checkpoints = th.load(MODEL_PATH)
    for i, actor in enumerate(maddpg.actors):
        actor_check = checkpoints['actor_%d' % (i)]
        actor.load_state_dict(checkpoints['actor_%d' % (i)])
        maddpg.actors_target[i] = deepcopy(actor)
    for i, critic in enumerate(maddpg.critics):
        critic.load_state_dict(checkpoints['critic_%d' % (i)])
        maddpg.critics_target[i] = deepcopy(critic)
    for i, actor_optim in enumerate(maddpg.actor_optimizer):
        actor_optim.load_state_dict(checkpoints['actor_optim_%d' % (i)])
    for i, critic_optim in enumerate(maddpg.critic_optimizer):
        critic_optim.load_state_dict(checkpoints['critic_optim_%d' % (i)])

prev_actor = None
prev_critic = None

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    # print("started 1")
    try:
        obs,pose = world.reset(random=True)
        pose = th.tensor(pose)
    except Exception as e:
        continue
    obs = np.stack(obs)
    # history initialization
    obs_t_minus_0 = copy(obs)
    obs_t_minus_1 = copy(obs)
    obs_t_minus_2 = copy(obs)
    obs_t_minus_3 = copy(obs)
    obs_t_minus_4 = copy(obs)
    obs_t_minus_5 = copy(obs)
    obs_history=np.zeros((n_agents,obs.shape[1]*6,obs.shape[2]))
    for i in range(n_agents):
        obs_history[i] = np.vstack((obs_t_minus_0[i],obs_t_minus_1[i],obs_t_minus_2[i],
                            obs_t_minus_3[i],obs_t_minus_4[i],obs_t_minus_5[i]))

    if isinstance(obs, np.ndarray):
        obs_history = th.from_numpy(obs_history).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    wrong_step = 0
    # print("started 2")
    num_steps = 0
    for t in range(max_steps):
        num_steps = num_steps + 1
        # print("test")
        # render every 100 episodes to speed up training
        if i_episode % 1 == 0 and e_render:
            world.render()
        obs_history = obs_history.type(FloatTensor)
        action_probs = maddpg.select_action(obs_history, pose).data.cpu()
        copied_tensor = action_probs.clone()
        action_probs_valid = np.copy(copied_tensor.numpy())
        action = []
        # for i,probs in enumerate(action_probs):
        #     rbt = world.robots[i]
        rbt = world.robots[0]
        # print(action_probs_valid[0])
        for j, frt in enumerate(rbt.get_frontiers()):
            if len(frt) == 0:
                # print(action_probs_valid[0][j])
                action_probs_valid[j] = 0
        action.append(categorical.Categorical(probs=th.tensor(action_probs_valid)).sample())

            # for j,frt in enumerate(rbt.get_frontiers()):
            #     if len(frt) == 0:
            #         print(action_probs_valid[i][j])
            #         action_probs_valid[i][j] = 0
            # action.append(categorical.Categorical(probs=th.tensor(action_probs_valid[i])).sample())

        action = th.tensor(onehot_from_action(action))
        acts = np.argmax(action,axis=1)
        for i in range(len(acts)):
            if len(world.robots[i].frontiers[acts[i]]) == 0:
                # NOOP 指令
                acts[i] = -1

        obs_, reward, done, _, next_pose = world.step(acts)
        next_pose = th.tensor(next_pose)
        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()

        obs_t_minus_5 = copy(obs_t_minus_4)
        obs_t_minus_4 = copy(obs_t_minus_3)
        obs_t_minus_3 = copy(obs_t_minus_2)
        obs_t_minus_2 = copy(obs_t_minus_1)
        obs_t_minus_1 = copy(obs_t_minus_0)
        obs_t_minus_0 = copy(obs_)
        obs_history_ = np.zeros((n_agents, obs.shape[1] * 6, obs.shape[2]))
        # for i in range(n_agents):
        #     obs_history_[i] = np.vstack((obs_t_minus_0[i], obs_t_minus_1[i], obs_t_minus_2[i],
        #                                      obs_t_minus_3[i], obs_t_minus_4[i], obs_t_minus_5[i]))
        obs_history_[0] = np.vstack((obs_t_minus_0[0], obs_t_minus_1[0], obs_t_minus_2[0],
                                             obs_t_minus_3[0], obs_t_minus_4[0], obs_t_minus_5[0]))

        if t != max_steps - 1:
            next_obs_history = th.tensor(obs_history_)
        elif done:
            next_obs_history = None
        else:
            next_obs_history = None
        total_reward += reward.sum()
        rr += reward.cpu().numpy()

        maddpg.memory.push(obs_history, action, next_obs_history, reward, pose, next_pose)
        obs_history = next_obs_history
        pose = next_pose
        if t % 5 == 0:
            # print("update policy")
            c_loss, a_loss = maddpg.update_policy()
            # print("update policy end")
        if done:
            break
    # print("rr", rr)
    num_step_file = open(os.getcwd() + '/../runs/' + time_now + '/num_steps.txt', "a")
    num_step_file.write("eps: " + str(i_episode) + " #step: " + str(num_steps) + "\n")

    num_step_file.close()


    # if not discard:
    maddpg.episode_done += 1
    if maddpg.episode_done % model_save_eps == 0:
        print('Save Models......')
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        dicts = {}

        for i in range(maddpg.n_agents):
            dicts['actor_%d' % (i)] = maddpg.actors_target[i].state_dict()
            dicts['critic_%d' % (i)] = maddpg.critics_target[i].state_dict()
            dicts['actor_optim_%d' % (i)] = maddpg.actor_optimizer[i].state_dict()
            dicts['critic_optim_%d' % (i)] = maddpg.critic_optimizer[i].state_dict()
        th.save(dicts, MODEL_DIR + '/model-%d.pth' % (maddpg.episode_done))
        # th.save(dicts, MODEL_DIR+'/model-%d.pth'%(config['robots']['number']))
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)
    # visual
    # writer.add_scalars('scalar/reward',{'total_rwd':total_reward,'r0_rwd':rr[0],'r1_rwd':rr[1]},i_episode)
    writer.add_scalars('scalar/reward', {'total_rwd': total_reward, 'r0_rwd': rr[0]}, i_episode)
    if i_episode > episodes_before_train and i_episode % 10 == 0:
        print(reward_record)
        mean_reward = th.mean(th.stack(reward_record[-100:]).cpu()).item()
        writer.add_scalars('scalar/mean_rwd', {'mean_reward': mean_reward}, i_episode)
        # writer.add_scalars('scalar/mean_rwd',{'mean_reward':np.mean(reward_record[-100:])}, i_episode)
    if not c_loss is None:
        # writer.add_scalars('loss/c_loss',{'r0':c_loss[0],'r1':c_loss[1]},i_episode)
        writer.add_scalars('loss/c_loss', {'r0': c_loss[0]}, i_episode)
    if not a_loss is None:
        # writer.add_scalars('loss/a_loss',{'r0':a_loss[0],'r1':a_loss[1]},i_episode)
        writer.add_scalars('loss/a_loss', {'r0': a_loss[0]}, i_episode)
    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')

world.close()
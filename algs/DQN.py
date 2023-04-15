import os
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BATCH_SIZE = 256
LR = 0.00001
GAMMA = 0.99
EPISILO = 0.2
EPISILO_MAX = 0.9
MEMORY_CAPACITY = 10000
Q_NETWORK_ITERATION = 50
MIN_EPISODE_BEFORE_TRAINING = 5


class Net(nn.Module):
    """docstring for Net"""
    def __init__(self, num_state, num_action):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_state, 512)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(512,512)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(512,num_action)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        q_vals = self.out(x)
        return q_vals


class DQN:
    max_grad_norm = 0.1
    reward_scaling = 1

    def __init__(self, state_dim, action_dim):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.eval_net, self.target_net = Net(self.state_dim, self.action_dim), Net(self.state_dim, self.action_dim)
        self.learn_step_counter = self.episode_counter = 0
        self.memory_counter = 0
        self.memory_full = False
        self.epsilon = EPISILO
        self.memory = np.zeros((MEMORY_CAPACITY, self.state_dim * 2 + 3))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.eval_net.cuda()
            self.target_net.cuda()

        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        self.BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor

    def choose_abstract_action(self, current_state, test_flag=False):
        current_state = (torch.unsqueeze(torch.tensor(current_state, dtype=torch.float), 0)).type(self.FloatTensor) # get a 1D array
        if np.random.rand() <= (self.epsilon if not test_flag else 1.0):    # greedy policy
            action_value = self.eval_net(current_state)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()[0]
        else:   # random policy
            action = random.randint(0, self.action_dim - 1)
        return action, None

    def store_transition(self, state, action, reward, next_state, done):
        mask = 0.0 if done else 1.0
        # print('state:')
        # print(state)
        # print(len(state))
        # print('[action,reward]')
        # print([action,reward])
        # print(len([action,reward]))
        # print('next_state')
        # print(next_state)
        # print(len(next_state))
        # print('[mask]')
        # print([mask])
        # print(len([mask]))

        transition = np.hstack((state, [action, reward], next_state, [mask]))

        self.memory[self.memory_counter, :] = transition
        self.memory_counter = (self.memory_counter + 1) % MEMORY_CAPACITY
        if self.memory_counter == 0:
            self.memory_full = True

    def learn(self, state, action, reward, next_state, action_prob, done, path_str=None):
        self.store_transition(state, action, reward * self.reward_scaling, next_state, done)

        if self.episode_counter < MIN_EPISODE_BEFORE_TRAINING:
            return

        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter = (self.learn_step_counter + 1) % Q_NETWORK_ITERATION
        self.epsilon = self.epsilon + (EPISILO_MAX - EPISILO) / (600 * 25000) if self.epsilon < EPISILO_MAX else EPISILO_MAX

        # sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY if self.memory_full else self.memory_counter, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = (torch.tensor(batch_memory[:, :self.state_dim], dtype=torch.float)).type(self.FloatTensor)
        batch_action = (torch.tensor(batch_memory[:, self.state_dim:self.state_dim+1], dtype=torch.long)).type(self.LongTensor)
        batch_reward = (torch.tensor(batch_memory[:, self.state_dim+1:self.state_dim+2], dtype=torch.float)).type(self.FloatTensor)
        batch_next_state = (torch.tensor(batch_memory[:, self.state_dim+2:2*self.state_dim+2], dtype=torch.float)).type(self.FloatTensor)
        batch_mask = (torch.tensor(batch_memory[:, 2*self.state_dim+2:2*self.state_dim+3], dtype=torch.float)).type(self.FloatTensor)

        # q_eval
        q_values = self.eval_net(batch_state)
        q_eval = q_values.gather(1, batch_action)
        max_action_indexes = q_values.detach().argmax(1)
        q_next = self.target_net(batch_next_state).detach().gather(1, max_action_indexes.unsqueeze(1))
        q_target = batch_reward + GAMMA * batch_mask * q_next
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if path_str:  # 存储loss
            aloss_file = open(os.path.join(os.path.join('./Outcome', path_str), 'agent_loss.txt'), 'a+')
            aloss_file.write(f'{float(loss)}\n')
            aloss_file.close()

    def save_param(self, path_str, i_episode):
        if i_episode % 10 == 0:
            torch.save(self.eval_net.state_dict(), os.path.join(os.path.join('./Model', path_str), f'agent-{i_episode}.pkl'))

        self.episode_counter = i_episode

import sys,os
import time

curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加路径到系统路径

from algs.DQN import DQN
from vanet.tools.models import AbstractCacheModel
from algs.baselines import LRU_Cache_Delegate, LFU_Cache_Delegate, GCP_Cache_Delegate, RC_Cache_Delegate
import torch
import numpy as np
import datetime
from algs.utils import plot_rewards
from algs.utils import save_results,make_dir
from algs.ppo2 import PPO
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
# class DRL_Veh_Cache_Delegate(AbstractCacheModel):
#     alg_name = 'drl_cache_model_delegate_for_vehicle'
#
#     def __init__(self, random_seed):
#         self.cache_algs = [LRU_Cache_Delegate(), LFU_Cache_Delegate(), GCP_Cache_Delegate(), RC_Cache_Delegate()]
#         self.cache_alg_chooser = DQN(state_dim=8, action_dim=len(self.cache_algs) + 1)
#         self.pre_state, self.pre_action, self.action_prob = None, None, None
#         self.random_seed = random_seed
#
#     def decision(self, obs):
#         # obs处理
#         action = self.cache_delegate(obs)
#         if action >= len(self.cache_algs):
#             return False, []
#         else:
#             flag, replaced_segs = self.cache_algs[action].decision(obs)
#             return flag, replaced_segs


class Config:
    def __init__(self) -> None:
        ################################## 环境超参数 ###################################
        self.algo_name = "PPO"  # 算法名称
        self.env_name = 'CartPole-v1'  # 环境名称
        self.continuous = False  # 环境是否为连续动作
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        # self.seed = 10  # 随机种子，置0则不设置随机种子
        self.train_eps = 200  # 训练的回合数
        self.test_eps = 20  # 测试的回合数
        ################################################################################

        ################################## 算法超参数 ####################################
        self.batch_size = 5  # mini-batch SGD中的批量大小
        self.gamma = 0.95  # 强化学习中的折扣因子
        self.n_epochs = 4
        self.actor_lr = 0.0003  # actor的学习率
        self.critic_lr = 0.0003  # critic的学习率
        self.gae_lambda = 0.95
        self.policy_clip = 0.2
        self.hidden_dim = 256
        self.update_fre = 20  # 策略更新频率
        ################################################################################

        ################################# 保存结果相关参数 ################################
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片
        ################################################################################
class DRL_Cache_Delegate(AbstractCacheModel):
    alg_name = 'drl_cache_model_delegate'

    def __init__(self, state_dim, path_str, node_type):
        self.cache_algs = [LRU_Cache_Delegate(), LFU_Cache_Delegate(), GCP_Cache_Delegate(), RC_Cache_Delegate()]
        self.cache_alg_chooser = DQN(state_dim=state_dim, action_dim=len(self.cache_algs) + 1)
        self.pre_state, self.pre_action, self.action_prob = None, None, None
        self.path_str = path_str
        self.node_type = node_type
        # self.cfg = Config()
        # self.agent = PPO(state_dim, 2, self.cfg)  # 创建智能体



    def decision(self, obs, test_flag):
        # obs处理
        state, pre_reward = obs['state'], obs['pre_reward']

        if self.pre_state is not None and not test_flag:
            self.cache_alg_chooser.learn(state=self.pre_state,
                                         action=self.pre_action,
                                         reward=pre_reward,
                                         next_state=state,
                                         action_prob=self.action_prob,
                                         done=obs['done'],
                                         path_str=os.path.join(self.path_str, self.node_type))
        action, a_log_prob = self.cache_alg_chooser.choose_abstract_action(state, test_flag=test_flag)
        self.pre_state, self.pre_action, self.action_prob = state, action, a_log_prob

        # 执行缓存决策
        if action >= len(self.cache_algs):
            return False, []
        else:
            flag, replaced_segs = self.cache_algs[action].decision(obs)
            return flag, replaced_segs

    def save(self, i_episode):
        self.cache_alg_chooser.save_param(path_str=os.path.join(self.path_str, self.node_type), i_episode=i_episode)

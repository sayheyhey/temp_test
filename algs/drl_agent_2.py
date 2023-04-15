import os
from algs.DQN import DQN
from vanet.tools.models import AbstractCacheModel
from algs.baselines import LRU_Cache_Delegate, LFU_Cache_Delegate, GCP_Cache_Delegate, RC_Cache_Delegate


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


class DRL_Cache_Delegate(AbstractCacheModel):
    alg_name = 'drl_cache_model_delegate'

    def __init__(self, state_dim, path_str, node_type):
        self.cache_algs = [LRU_Cache_Delegate(), LFU_Cache_Delegate(), GCP_Cache_Delegate(), RC_Cache_Delegate()]
        self.cache_alg_chooser = DQN(state_dim=state_dim, action_dim=len(self.cache_algs) + 1)
        self.pre_state, self.pre_action, self.action_prob = None, None, None
        self.path_str = path_str
        self.node_type = node_type

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

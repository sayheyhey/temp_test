import collections
import numpy as np
import vanet.env_params as p
import sys, os
import time

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import gym
import torch
import numpy as np
import datetime
from algs.utils import plot_rewards
from algs.utils import save_results, make_dir
from algs.ppo2 import PPO
import time

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class Config:
    def __init__(self) -> None:
        ################################## 环境超参数 ###################################
        self.algo_name = "PPO"  # 算法名称
        self.env_name = 'test'  # 环境名称
        self.continuous = False  # 环境是否为连续动作
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 检测GPU
        ################################################################################

        ################################## 算法超参数 ####################################
        self.batch_size = 5  # mini-batch SGD中的批量大小
        self.gamma = 0.95  # 强化学习中的折扣因子
        self.n_epochs = 4
        self.actor_lr = 0.0003  # actor的学习率
        self.critic_lr = 0.0003  # critic的学习率
        self.gae_lambda = 0.95
        self.policy_clip = 0.2
        self.hidden_dim = 512
        self.update_fre = 20  # 策略更新频率
        ################################################################################

        ################################# 保存结果相关参数 ################################
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片


"""
Cache Models
"""


class Request:
    def __init__(self, r_id, origin_time, content_id, vehicle_id, segment_units, segment_unit_size):
        self.r_id = r_id  # id of the request
        self.origin_time = origin_time  # launched time of the request
        self.content_id = content_id  # id of the requested content
        self.vehicle_id = vehicle_id  # id of the vehicle that launches the request
        self.left_segment_list = [segment_unit_size] * segment_units  # unprocessed data amount of each segment
        # self.handled = False  # indicator of whether the request is successfully completed
        self.last_timeslot_segment_provider = None  # node that communicate to vehicle_id during the last time slot
        self.local_cache_hit_segment_cnt = 0
        self.v2v_cache_hit_segment_cnt = 0
        self.v2r_cache_hit_segment_cnt = 0
        self.mbs_hit_segment_cnt = 0
        self.finish_time = -1


class RSU_Request:
    def __init__(self, rsu_id, origin_time, finish_time, content_id, seg_id):
        self.rsu_id = rsu_id
        self.origin_time = origin_time
        self.finish_time = finish_time
        self.content_id = content_id
        self.seg_id = seg_id


class Content:
    def __init__(self, c_id, popularity, segment_units, segment_unit_size):
        self.id = c_id  # content id
        self.popularity = popularity  # content popularity (request probability)
        self.segment_units = segment_units  # content size (segment unit)
        self.segment_unit_size = segment_unit_size


class Segment(object):
    def __init__(self, content_id, segment_id, size, data) -> None:
        self.id = content_id * 10 + segment_id
        self.contentId = content_id
        self.segmentId = segment_id
        self.size = size

        self.frequency = 1
        self.popularity = data['popularity']
        self.time = data['now_time']
        self.call_log = [self.time]


# 超类的作用：根据新来的seg判断是否替换，如何替换


class AbstractCacheModel(object):

    def __init__(self) -> None:
        pass

    def decision(self, obs):  # newSeg = id, type:string
        # 该seg存在
        # 不存在，且放得下
        # 不存在，且放不下
        raise NotImplementedError

    def save(self, *params):
        pass


"""
Node Models
"""


class Node:
    type = 'Node'

    def __init__(
            self,
            node_id: int,
            x: float,
            y: float,
            cache_delegate: AbstractCacheModel = None,
            s: float = float('inf')
    ):
        self.id = node_id
        self.x, self.y = x, y
        self.cache_size = s
        self.cache_delegate = cache_delegate
        self.cache_content_segments_set = dict()  # {1: set(1, 2, 4)]} means this node caches seg 1,2,4 of content 1
        self.buffer = dict()  # 与cache delegate对接
        self.left_capacity = s

        self.neighbor_vehicles = []
        self.neighbor_rsus = []

        self.cache_hit_dict = collections.defaultdict(dict)
        self.local_request_cnt_dict = collections.defaultdict(dict)
        self.remote_request_cnt_dict = collections.defaultdict(dict)
        self.unable_request_cnt_dict = collections.defaultdict(dict)

        self.cur_request_cnt = 0
        self.pre_decision_time = 0

        # self.cache_hit_cnt_in_decision_interval = 0  # reward
        self.cache_hit_size_in_decision_interval = 0
        # self.step_count = 0
        self.ep_reward = 0
        self.reward = []
        self.ma_rewards = []

    def debug_print(self):
        # with open('model_buffer.txt', 'a') as f:
        #     print(self.cache_content_segments_set, file=f)
        # f.close()
        pass
    def update_cache_properties(self, content_id, seg_id, seg_size, nowTime, popularity):
        seg_id = content_id * 10 + seg_id
        data = dict()
        data['popularity'] = popularity
        data['now_time'] = nowTime
        self.buffer[seg_id]=Segment(content_id, seg_id, seg_size, data)
        self.buffer[seg_id].frequency += 1
        self.buffer[seg_id].time = nowTime
        self.buffer[seg_id].Pr = popularity
        self.buffer[seg_id].call_log.append(nowTime)

    def increment_cache_hit_size_and_cnt(self, timer, content_id, size_amount, cnt=1):
        if content_id not in self.cache_hit_dict[timer]:
            self.cache_hit_dict[timer][content_id] = 0
        self.cache_hit_dict[timer][content_id] += cnt
        # self.cache_hit_cnt_in_decision_interval += cnt
        self.cache_hit_size_in_decision_interval += cnt

    def increment_local_request_cnt(self, timer, content_id, cnt=1):
        if content_id not in self.local_request_cnt_dict[timer]:
            self.local_request_cnt_dict[timer][content_id] = 0
        self.local_request_cnt_dict[timer][content_id] += cnt

    def increment_remote_request_cnt(self, timer, content_id, cnt=1):
        if content_id not in self.remote_request_cnt_dict[timer]:
            self.remote_request_cnt_dict[timer][content_id] = 0
        self.remote_request_cnt_dict[timer][content_id] += cnt

    def increment_unable_request_cnt(self, timer, content_id, cnt=1):
        if content_id not in self.unable_request_cnt_dict[timer]:
            self.unable_request_cnt_dict[timer][content_id] = 0
        self.unable_request_cnt_dict[timer][content_id] += cnt

    def update_adjacent_vehicles(self, neighbor_vehicles):
        self.neighbor_vehicles = neighbor_vehicles

    def update_adjacent_rsus(self, neighbor_rsus):
        self.neighbor_rsus = neighbor_rsus

    def get_drl_state(self, obs):
        node_features, seg_features, env_features, history_features = self.get_node_features(), self.get_seg_features(
            obs), self.get_env_features(obs), self.get_history_features(obs)
        obs['state'] = np.concatenate((node_features, seg_features, env_features, history_features), axis=0)
        # obs['pre_reward'] = self.cache_hit_cnt_in_decision_interval
        obs['pre_reward'] = self.cache_hit_size_in_decision_interval
        return obs

    def get_node_features(self):
        # node_features = [self.x, self.y, self.cache_size, self.left_capacity, self.cur_request_cnt]

        if self.type == 'Vehicle':
            # 归一化坐标、容量、请求
            node_features = [self.x, self.y, self.left_capacity, self.cur_request_cnt]
            node_features[0] = (node_features[0] - p.VEH_X_MIN) / (p.VEH_X_MAX - p.VEH_X_MIN)
            node_features[1] = (node_features[1] - p.VEH_Y_MIN) / (p.VEH_Y_MAX - p.VEH_Y_MIN)
            node_features[2] = node_features[2] / p.VEH_CACHE_SIZE
            node_features[3] = node_features[3] / p.VEH_CUR_REQUEST_CNT_MAX
        else:
            node_features = [self.x, self.y, self.left_capacity]
            node_features[0] = (node_features[0] - p.RSU_X_MIN) / (p.RSU_X_MAX - p.RSU_X_MIN)
            node_features[1] = (node_features[1] - p.RSU_Y_MIN) / (p.RSU_Y_MAX - p.RSU_Y_MIN)
            node_features[2] = node_features[2] / p.RSU_CACHE_SIZE

        # 缓存空间配置
        cache_features = [0] * p.CONTENT_LIBRARY_SIZE
        for content_id in self.cache_content_segments_set.keys():
            cache_features[content_id] = len(self.cache_content_segments_set[content_id]) / p.ENCODED_SEGMENTS_UNITS_MAX

        return node_features + cache_features

    def get_seg_features(self, obs):
        seg_data = obs['seg_size']
        seg_units = obs['seg_unit']
        seg_popularity = obs['static_popularity']
        # seg_existence = int(
        #     obs['content_id'] in self.cache_content_segments_set and obs['seg_id'] in self.cache_content_segments_set[
        #         int(obs['content_id'])])
        seg_content_cnt = len(self.cache_content_segments_set[obs['content_id']]) if obs[
                                                                                         'content_id'] in self.cache_content_segments_set else 0
        seg_features = [seg_data, seg_units, seg_content_cnt, seg_popularity]

        # feature normarlization
        seg_features[0] = (seg_features[0] - p.SEGMENT_UNIT_SIZE_MIN) / (
                    p.SEGMENT_UNIT_SIZE_MAX - p.SEGMENT_UNIT_SIZE_MIN)
        seg_features[1] = (seg_features[1] - p.ENCODED_SEGMENTS_UNITS_MIN) / (
                    p.ENCODED_SEGMENTS_UNITS_MAX )
        seg_features[2] = seg_content_cnt / p.ENCODED_SEGMENTS_UNITS_MAX

        return seg_features

    def get_env_features(self, obs):
        decision_interval = obs['nowTime'] - self.pre_decision_time
        new_seg_id, new_content_id = obs['content_id'] * 10 + obs['seg_id'], obs['content_id']
        neighbor_vehs_cnt = len(self.neighbor_vehicles) if self.neighbor_vehicles is not None else 0
        # neighbor_rsus_cnt = len(self.neighbor_rsus) if self.neighbor_rsus is not None else 0
        diff_seg_cnt = diff_content_cnt = new_seg_cache_cnt = new_content_cache_cnt = 0
        seg_vis, content_vis = set(), set()

        def collect_content_and_seg_cnt(node):
            nonlocal diff_seg_cnt, diff_content_cnt, new_content_cache_cnt, new_seg_cache_cnt
            for cache_content_id in node.cache_content_segments_set.keys():
                if cache_content_id not in content_vis:
                    content_vis.add(cache_content_id)
                    diff_content_cnt += 1
                if cache_content_id == new_content_id:
                    new_content_cache_cnt += 1
                for seg_index in node.cache_content_segments_set[cache_content_id]:
                    cache_seg_id = cache_content_id * 10 + seg_index
                    if cache_seg_id not in seg_vis:
                        seg_vis.add(cache_seg_id)
                        diff_seg_cnt += 1
                    if cache_seg_id == new_seg_id:
                        new_seg_cache_cnt += 1

        # 统计所以有的content和seg的缓存命中
        for veh in self.neighbor_vehicles:
            collect_content_and_seg_cnt(veh)

        # for rsu in self.neighbor_rsus:
        #     collect_content_and_seg_cnt(rsu)

        if self.type == 'Vehicle':
            return [decision_interval / p.DECISION_INTERVAL_MAX,
                    # neighbor_rsus_cnt / p.NEIGHBOR_RSU_CNT_MAX,
                    neighbor_vehs_cnt / p.NEIGHBOR_VEH_CNT_MAX,
                    diff_content_cnt / p.CONTENT_LIBRARY_SIZE,
                    diff_seg_cnt / (p.CONTENT_LIBRARY_SIZE * p.ENCODED_SEGMENTS_UNITS_MAX),
                    new_seg_cache_cnt / p.NEW_SEG_CACHE_CNT_MAX,
                    new_content_cache_cnt / p.NEW_CONTENT_CACHE_CNT_MAX]
        else:
            return [decision_interval / p.DECISION_INTERVAL_MAX,
                    neighbor_vehs_cnt / p.NEIGHBOR_VEH_CNT_MAX,
                    diff_content_cnt / p.CONTENT_LIBRARY_SIZE,
                    diff_seg_cnt / (p.CONTENT_LIBRARY_SIZE * p.ENCODED_SEGMENTS_UNITS_MAX),
                    new_seg_cache_cnt / p.NEW_SEG_CACHE_CNT_MAX,
                    new_content_cache_cnt / p.NEW_CONTENT_CACHE_CNT_MAX]

    def get_history_features(self, obs):
        cache_hit_cnt_list, new_content_cache_hit_cnt_list = [0] * p.SLIDING_WINDOW_SIZE, [0] * p.SLIDING_WINDOW_SIZE
        local_request_cnt_list, new_content_local_request_cnt_list = [0] * p.SLIDING_WINDOW_SIZE, [
            0] * p.SLIDING_WINDOW_SIZE
        remote_request_cnt_list, new_content_remote_request_cnt_list = [0] * p.SLIDING_WINDOW_SIZE, [
            0] * p.SLIDING_WINDOW_SIZE
        unable_request_cnt_list, new_content_unable_request_cnt_list = [0] * p.SLIDING_WINDOW_SIZE, [
            0] * p.SLIDING_WINDOW_SIZE

        def update_history_info_in_sliding_window(history_info_dict, res_list, new_content_res_list):
            tmp_dict = {}
            tmp_most_content_id = None

            for t in range(p.SLIDING_WINDOW_SIZE, 0, -1):
                for content_id in history_info_dict[int(obs['nowTime'] - t)].keys():
                    res_list[p.SLIDING_WINDOW_SIZE - t] += history_info_dict[int(obs['nowTime'] - t)][content_id]
                    if content_id == obs['content_id']:
                        new_content_res_list[p.SLIDING_WINDOW_SIZE - t] += history_info_dict[int(obs['nowTime'] - t)][
                            content_id]

                    if content_id not in tmp_dict:
                        tmp_dict[content_id] = 0
                    tmp_dict[content_id] += 1

                    if tmp_most_content_id is None or tmp_dict[tmp_most_content_id] < tmp_dict[content_id]:
                        tmp_most_content_id = content_id

            return tmp_most_content_id is not None and tmp_most_content_id == obs['content_id']

        cache_hit_flag = update_history_info_in_sliding_window(self.cache_hit_dict, cache_hit_cnt_list,
                                                               new_content_cache_hit_cnt_list)
        local_flag = update_history_info_in_sliding_window(self.local_request_cnt_dict, local_request_cnt_list,
                                                           new_content_local_request_cnt_list)
        remote_flag = update_history_info_in_sliding_window(self.remote_request_cnt_dict, remote_request_cnt_list,
                                                            new_content_remote_request_cnt_list)
        unable_flag = update_history_info_in_sliding_window(self.unable_request_cnt_dict, unable_request_cnt_list,
                                                            new_content_unable_request_cnt_list)

        if self.type == 'Vehicle':
            return [int(cache_hit_flag), int(local_flag), int(remote_flag), int(
                unable_flag)] + [i / p.CACHE_HIT_CNT_MAX for i in cache_hit_cnt_list] + [i / p.CACHE_HIT_CNT_MAX for i
                                                                                         in
                                                                                         new_content_cache_hit_cnt_list] + [
                       i / p.LOCAL_REQUEST_MAX for i in local_request_cnt_list] + [i / p.LOCAL_REQUEST_MAX for i in
                                                                                   new_content_local_request_cnt_list] + [
                       i / p.REMOTE_REQUEST_MAX for i in remote_request_cnt_list] + [i / p.REMOTE_REQUEST_MAX for i in
                                                                                     new_content_remote_request_cnt_list] + [
                       i / p.UNABLE_REQUEST_MAX for i in unable_request_cnt_list] + [
                       i / p.NEW_CONTENT_UNABLE_REQUEST_MAX for i in new_content_unable_request_cnt_list]
        else:
            assert self.type == 'RSU', 'Invalid node type'

            return [int(cache_hit_flag), int(local_flag), int(remote_flag), int(
                unable_flag)] + [i / p.CACHE_HIT_CNT_MAX for i in cache_hit_cnt_list] + [i / p.CACHE_HIT_CNT_MAX for i
                                                                                         in
                                                                                         new_content_cache_hit_cnt_list] + [
                       i / p.REMOTE_REQUEST_MAX for i in remote_request_cnt_list] + [i / p.REMOTE_REQUEST_MAX for i in
                                                                                     new_content_remote_request_cnt_list] + [
                       i / p.UNABLE_REQUEST_MAX for i in unable_request_cnt_list] + [
                       i / p.NEW_CONTENT_UNABLE_REQUEST_MAX for i in new_content_unable_request_cnt_list]

    def update_cache_segments(self, obs, test_flag):
        content_id, seg_id, seg_size = obs['content_id'], obs['seg_id'], obs['seg_size']
        if content_id * 10 + seg_id in self.buffer:
            return
        # 如果该seg没在buffer里面，表示为新的seg
        seg_data = {'now_time': obs['nowTime'], 'popularity': obs['popularity']}
        step_count = obs['req_id']+1
        new_seg = Segment(content_id, seg_id, seg_size, seg_data)
        obs['newSeg'], obs['buffer'], obs['left_capacity'] = new_seg, self.buffer, self.left_capacity
        # 获取当前的obsation
        if self.cache_delegate.alg_name != 'LRU' and self.cache_delegate.alg_name !='LFU' and self.cache_delegate.alg_name !='RC':
            obs = self.get_drl_state(obs)
            cfg = Config()
            state = obs['state']

            # print(f'state:{state}')
            state_dim = len(state)
            # print(state_dim)
            agent = PPO(state_dim, 2, cfg)  # 创建智能体
            action, probs, value = agent.choose_action(state)
            # reward = self.get_reward(obs, action)
            # print(f'reward:{reward}')
            reward = self.cache_decision(obs, action)
            obs = self.get_drl_state(obs)
            # state_ = obs['state']
            # print(f'state_:{state_}')
            done = obs['done']
            # print(f'done:{done}')
            # reward = obs['pre_reward']
            if test_flag:
                with open('./Outcome/test/DRL-DRL/seed1/reward.txt', 'a+') as f:
                    f.write(f'{reward}'+' ')
            agent.memory.push(state, action, probs, value, reward, done)
            if not test_flag:
                if step_count % cfg.update_fre == 0 :
                    agent.update()
            self.cache_hit_size_in_decision_interval = 0
            self.pre_decision_time = obs['nowTime']

        '''
        在这一步做决策：根据信息决定是否替换内容
        state = obs['state']  可以修改obs的生成
        1)action, prob, val = self.agent.choose_action(state) 
        >>>agent怎么处理？？？
        >>>drl_agent中生成agent对象（未完）
        print(action)
        2)决定是否进行替换
        编写一个替换策略
        3）更新state,reward,done
        4）
        5）
        6）
        '''
        if self.cache_delegate.alg_name == 'LFU' or self.cache_delegate.alg_name == 'LRU' or self.cache_delegate.alg_name =='RC':
            obs = self.get_drl_state(obs)
            replace_flag, replaced_segs = self.cache_delegate.decision(obs, test_flag)
            if replace_flag:

                # 替换旧cache segments
                for old_seg in replaced_segs:
                    old_content_id, old_seg_id = old_seg.contentId, old_seg.segmentId
                    #
                    # assert old_content_id in self.cache_content_segments_set, f'replace nonexistent content {old_content_id} on vehicle {self.id}'
                    # assert old_seg_id in self.cache_content_segments_set[
                    #     old_content_id], f'replace nonexistent segment {old_seg_id} of content {old_content_id} on vehicle {self.id}'
                    # # print(f'{self.type}节点{self.id}删除了之前缓存的content{old_content_id}的segment{old_seg_id}')
                    if old_content_id in self.cache_content_segments_set.keys():
                        del self.cache_content_segments_set[old_content_id]
                        self.left_capacity += old_seg.size

                    # if len(self.cache_content_segments_set[old_content_id]) == 0:
                    #     del self.cache_content_segments_set[old_content_id]
                # cache中添加新segment
                if content_id not in self.cache_content_segments_set and self.left_capacity > seg_size:
                    self.cache_content_segments_set[content_id] = set()
                    self.cache_content_segments_set[content_id].add(seg_id)
                    self.left_capacity -= seg_size
                # print(f'{self.type}节点{self.id}缓存了content{content_id}的segment{seg_id}')


                assert self.left_capacity >= 0, 'self.left_capacity < 0'

                now_time = obs['nowTime']
                output_str = f'时刻{now_time}处, {self.type}节点{self.id}的缓存包括: '
                for c_i in list(self.cache_content_segments_set.keys()):
                    for segment_i in self.cache_content_segments_set[c_i]:
                        output_str += f'content{c_i}的segment{segment_i}  '

            # self.debug_print()
            # self.cache_delegate.buffer_display()

            #重置reward counter


    def cache_decision(self, obs, action):
        content_id = obs['content_id']
        seg_size = obs['seg_size']
        seg_id = obs['seg_id']
        reward = 0
        if obs['content_id'] in self.cache_content_segments_set.keys():
            x_state = 1
        else:
            x_state = 0
        # content_cache = self.cache_content_segments_set   # {1: set(1, 2, 4)]}
        # 第一种情况：如果做出的决定是不缓存，而自身已经缓存，则要进行删除
        if action == 0 and x_state == 1:
            # 计算到目前为止所有的缓存命中数目
            all_hit = 0
            for time in self.local_request_cnt_dict.keys():
                for id in self.local_request_cnt_dict[time].keys():
                    all_hit += self.local_request_cnt_dict[time][id]
            # 计算该content的缓存命中数目
            content_id_hit = 0
            for time in self.local_request_cnt_dict.keys():
                if content_id in self.local_request_cnt_dict[time].keys():
                    content_id_hit += self.local_request_cnt_dict[time][content_id]

            reward = -content_id_hit/all_hit

            del self.cache_content_segments_set[content_id]
            self.left_capacity += seg_size
            # if all_hit:
            #     reward = -content_id_hit/all_hit
            # else:
            #     reward = 0

        # 第二种情况：如果做出的决定是缓存，而自己没有缓存，则进行缓存
            # 考虑还有剩余的足够空间以及当空间已经满的时候，进行替换
        if action == 1 and x_state == 0:
            # 计算所有时刻的缓存命中数目
            all_hit = 0
            for time in self.local_request_cnt_dict.keys():
                for id in self.local_request_cnt_dict[time].keys():
                    all_hit += self.local_request_cnt_dict[time][id]
            # 计算所有时刻该content的缓存命中数目
            content_id_hit = 0
            for time in self.local_request_cnt_dict.keys():
                if content_id in self.local_request_cnt_dict[time].keys():
                    content_id_hit += self.local_request_cnt_dict[time][content_id]

            old_contetn_id_hit = 0
            # 有足够空间进行缓存，不需要替换
            if self.left_capacity >= obs['seg_size']:
                self.cache_content_segments_set[content_id] = set()
                self.cache_content_segments_set[content_id].add(seg_id)
                self.left_capacity -= seg_size
                del_list = []

            #  需要进行替换
            else:
                newSeg, buffer, left_capacity = obs['newSeg'], obs['buffer'], obs['left_capacity']
                del_list = []
                sorted_buffer = sorted(buffer.values(), key=lambda x: x.frequency)
                for seg in sorted_buffer:
                    if left_capacity >= newSeg.size:
                        break
                    # print(f'seg.id{seg_id}')
                    # print(f'buffer{buffer}')

                    # 进行替换操作
                    if seg.id in buffer.keys() and (self.left_capacity+seg.size)>newSeg.size:
                        self.left_capacity +=seg.size
                        self.left_capacity -=newSeg.size
                        self.cache_content_segments_set[content_id]=set()
                        self.cache_content_segments_set[content_id].add(newSeg.id)
                        for time in self.local_request_cnt_dict.keys():
                            if seg.contentId in self.local_request_cnt_dict[time].keys():
                                old_contetn_id_hit += self.local_request_cnt_dict[time][seg.contentId]
                        del buffer[seg.id]
                        del_list.append(seg)
                        buffer[newSeg.id] = newSeg
                        break
            reward = (content_id_hit - old_contetn_id_hit)/all_hit
            # if all_hit:
            #     reward = (content_id_hit-old_contetn_id_hit)/all_hit
            # else:
            #     reward = 0

        # 第三种情况
        if action == x_state:
            reward = 0
        obs['left_capacity'] = self.left_capacity
        return reward

    def get_reward(self, obs, action):
        # del_list_len =len(del_list)
        content_id = obs['content_id']
        seg_size = obs['seg_size']
        seg_id = obs['seg_id']
        if obs['content_id'] in self.cache_content_segments_set.keys():
            x_state = 1
        else:
            x_state = 0
        # 获取缓存命中率信息
        # decision_interval = obs['nowTime'] - self.pre_decision_time
        new_seg_id, new_content_id = obs['content_id'] * 10 + obs['seg_id'], obs['content_id']
        neighbor_vehs_cnt = len(self.neighbor_vehicles) if self.neighbor_vehicles is not None else 0
        diff_seg_cnt = diff_content_cnt = new_seg_cache_cnt = new_content_cache_cnt = 0
        seg_vis, content_vis = set(), set()
        abstract_content_cnt = {}
        old_content_cnt = 0

        def collect_content_and_seg_cnt(node):
            nonlocal diff_seg_cnt, diff_content_cnt, new_content_cache_cnt, new_seg_cache_cnt
            for cache_content_id in node.cache_content_segments_set.keys():
                if cache_content_id not in content_vis:
                    content_vis.add(cache_content_id)
                    # 所有node缓存的不同的内容数目
                    diff_content_cnt += 1
                if cache_content_id == new_content_id:
                    new_content_cache_cnt += 1
                    if new_content_id not in abstract_content_cnt.keys():
                        abstract_content_cnt[new_content_id] = new_content_cache_cnt
                    else:
                        abstract_content_cnt[new_content_id] += 1
                for seg_index in node.cache_content_segments_set[cache_content_id]:
                    cache_seg_id = cache_content_id * 10 + seg_index
                    if cache_seg_id not in seg_vis:
                        seg_vis.add(cache_seg_id)
                        diff_seg_cnt += 1
                    if cache_seg_id == new_seg_id:
                        new_seg_cache_cnt += 1

        # 统计所以有的content和seg的缓存命中
        for veh in self.neighbor_vehicles:
            collect_content_and_seg_cnt(veh)
        all_cnt = 0
        for key in abstract_content_cnt.keys():
            all_cnt += abstract_content_cnt[key]
        if action == 0 and x_state == 1:
            reward = (all_cnt - abstract_content_cnt[content_id]) / p.NEW_SEG_CACHE_CNT_MAX
        elif action == 1 and x_state == 0:
            if self.left_capacity >= obs['seg_size']:
                del_list = []
            else:
                newSeg, buffer, left_capacity = obs['newSeg'], obs['buffer'], obs['left_capacity']
                del_list = []

                for id, seg in buffer.items():  # 去除时间差超过S_INTERVAL的访问
                    for i in range(0, len(seg.call_log)):
                        if seg.call_log[i] + self.interval >= newSeg.time:
                            seg.call_log = seg.call_log[i:]
                            seg.frequency -= i
                            break

                sorted_buffer = sorted(buffer.values(), key=lambda x: x.frequency)
                for seg in sorted_buffer:
                    if left_capacity >= newSeg.size:
                        break
                    del buffer[seg.id]
                    left_capacity += seg.size
                    del_list.append(seg)
            if del_list == []:
                reward = 0
            else:
                for old_seg in del_list:
                    old_content_id, old_seg_id = old_seg.contentId, old_seg.segmentId
                    old_content_cnt += abstract_content_cnt[old_content_id]
                reward = (all_cnt + abstract_content_cnt[content_id] - old_content_cnt / len(
                    del_list)) / p.NEW_SEG_CACHE_CNT_MAX
        else:
            reward = 1
        return reward


class Vehicle(Node):
    type = 'Vehicle'

    def __init__(self, node_id, x, y, cache_delegate, s):
        super(Vehicle, self).__init__(node_id, x, y, cache_delegate, s)
        assert self.cache_delegate is not None, f'cache delegate of vehicle {self.id} is None'
        self.in_comm_flag = False
        self.in_comm_time_per_slot = 0

    def refresh_in_comm_time_per_slot(self):
        self.in_comm_time_per_slot = 0

    def update_location(self, x, y):
        self.x, self.y = x, y


class RSU(Node):
    type = 'RSU'

    def __init__(self, node_id, x, y, cache_delegate, s):
        super(RSU, self).__init__(node_id, x, y, cache_delegate, s)
        assert self.cache_delegate is not None, f'cache delegate of rsu {self.id} is None'
        self.seg_miss_cnt = 0


class MBS(Node):
    type = 'MBS'

    def update_cache_properties(self, *params):
        pass

    def update_cache_segments(self, *params):
        pass


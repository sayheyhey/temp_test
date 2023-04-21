import os
import random
import logging
import argparse
import numpy as np
import torch as th
from pathlib import Path
import vanet.env_params as p
from vanet.vanet_env import Env
from Pathfinder import my_mkdir
from algs.drl_agent import DRL_Cache_Delegate
from algs.baselines import LRU_Cache_Delegate, LFU_Cache_Delegate, GCP_Cache_Delegate, RC_Cache_Delegate
# from algs.baselines import LRU_Cache_Delegate, LFU_Cache_Delegate, GCP_Cache_Delegate, RC_Cache_Delegate, GDSF_Cache_Delegate


parser = argparse.ArgumentParser(description='experiment setting')
parser.add_argument('--env_id', default='test', type=str)
parser.add_argument('--training_episode_count', default=20000, type=int)
parser.add_argument('--drl_agent_test_interval', default=10, type=int)
# parser.add_argument('--iteration_count', default=600, type=int)
parser.add_argument('--random_seed', default=1, type=int)
parser.add_argument('--vehicle_cache_alg', default='DRL', type=str)
parser.add_argument('--rsu_cache_alg', default='DRL', type=str)
parser.add_argument('--rsu_cache_enabled_flag', default=False, type=bool)
parser.add_argument('--vehicle_cache_enabled_flag', default=True, type=bool)

args = parser.parse_args()
logging.basicConfig(level=logging.WARNING)

# cache delegate list
vehicle_cache_alg_index = [i for i, k in enumerate(p.CACHE_ALGS) if args.vehicle_cache_alg in k][0]
rsu_cache_alg_index = [i for i, k in enumerate(p.CACHE_ALGS) if args.rsu_cache_alg in k][0]

my_mkdir('/Outcome/baselines')

# 生成DRL文件目录
root_dir_path = None
if vehicle_cache_alg_index == len(p.CACHE_ALGS) - 1 or rsu_cache_alg_index == len(p.CACHE_ALGS) - 1:
    root_dir_path = f'{args.env_id}/{p.CACHE_ALGS[vehicle_cache_alg_index]}-{p.CACHE_ALGS[rsu_cache_alg_index]}/seed{args.random_seed}'
    outcome_dir_path = os.path.join('/Outcome', root_dir_path)
    if (Path(os.path.join('./Outcome', root_dir_path))).is_dir():
        print('Already done exps')
        exit(-1)
    if vehicle_cache_alg_index == len(p.CACHE_ALGS) - 1:
        my_mkdir(os.path.join('/Outcome', os.path.join(root_dir_path, 'Vehicle')))
        my_mkdir(os.path.join('/Model', os.path.join(root_dir_path, 'Vehicle')))
    if rsu_cache_alg_index == len(p.CACHE_ALGS) - 1:
        my_mkdir(os.path.join('/Outcome', os.path.join(root_dir_path, 'RSU')))
        my_mkdir(os.path.join('/Model', os.path.join(root_dir_path, 'RSU')))
    my_mkdir(outcome_dir_path)

vehicle_cache_delegate_list = [LRU_Cache_Delegate(), LFU_Cache_Delegate(), GCP_Cache_Delegate(), RC_Cache_Delegate(), DRL_Cache_Delegate(state_dim=218+8*p.SLIDING_WINDOW_SIZE, path_str=root_dir_path, node_type='Vehicle')]
rsu_cache_delegate_list = [LRU_Cache_Delegate(), LFU_Cache_Delegate(), GCP_Cache_Delegate(), RC_Cache_Delegate(), DRL_Cache_Delegate(state_dim=217+6*p.SLIDING_WINDOW_SIZE, path_str=root_dir_path, node_type='RSU')]
# vehicle_cache_delegate_list = [LRU_Cache_Delegate(), LFU_Cache_Delegate(), GCP_Cache_Delegate(), RC_Cache_Delegate(), GDSF_Cache_Delegate(), DRL_Veh_Cache_Delegate()]
# rsu_cache_delegate_list = [LRU_Cache_Delegate(), LFU_Cache_Delegate(), GCP_Cache_Delegate(), RC_Cache_Delegate(), GDSF_Cache_Delegate(), DRL_Rsu_Cache_Delegate()]


# seed fix
def setup_seed(random_seed):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    th.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(random_seed)
        th.cuda.manual_seed(random_seed)


setup_seed(args.random_seed)
print(f"torch cuda available={th.cuda.is_available()}")

overall_cache_delegates_list = [vehicle_cache_delegate_list[vehicle_cache_alg_index], rsu_cache_delegate_list[rsu_cache_alg_index]]

# initialize the environment
env = Env(env_id=args.env_id, cache_delegates_list=overall_cache_delegates_list, rsu_cache_enabled=args.rsu_cache_enabled_flag, veh_cache_enabled=args.vehicle_cache_enabled_flag)

for episode_i in range(1, args.training_episode_count+1):
    print(f'第{episode_i}轮训练')

    # reset the environment
    env.reset()
    while True:
        done = env.step()
        if done:
            break

    # result output
    all_requests_res = env.get_all_request_process_result()
    success_request_cnt = failed_request_cnt = 0
    mbs_hit_segment_cnt = local_cache_hit_segment_cnt = v2v_cache_hit_segment_cnt = v2r_cache_hit_segment_cnt = 0
    total_success_request_handle_duration = 0
    for req in all_requests_res:
        mbs_hit_segment_cnt += req.mbs_hit_segment_cnt
        local_cache_hit_segment_cnt += req.local_cache_hit_segment_cnt
        v2v_cache_hit_segment_cnt += req.v2v_cache_hit_segment_cnt
        v2r_cache_hit_segment_cnt += req.v2r_cache_hit_segment_cnt
        if req.finish_time != -1:
            success_request_cnt += 1
            total_success_request_handle_duration += (req.finish_time - req.origin_time)
        else:
            failed_request_cnt += 1

    total_cache_hit_cnt = local_cache_hit_segment_cnt + v2v_cache_hit_segment_cnt + v2r_cache_hit_segment_cnt
    res_succ_request_ratio = (success_request_cnt / (success_request_cnt + failed_request_cnt)) if success_request_cnt > 0 else 0
    res_cache_hit_ratio = (total_cache_hit_cnt / (mbs_hit_segment_cnt + total_cache_hit_cnt)) if total_cache_hit_cnt > 0 else 0
    res_avg_response_time = (total_success_request_handle_duration / success_request_cnt) if success_request_cnt > 0 else 0
    print(f'Successful Handled Request Ratio: {res_succ_request_ratio}')
    print(f'\t成功处理的content requests共{success_request_cnt}个\t处理失败的content requests共{failed_request_cnt}个')
    print(f'Cache Hit Ratio: {res_cache_hit_ratio}')
    print(f'\tvehicle本地缓存服务的segments共{local_cache_hit_segment_cnt}个\tV2V缓存服务的segments共{v2v_cache_hit_segment_cnt}个\tV2R缓存服务的segments共{v2r_cache_hit_segment_cnt}个\tMBS服务的segments共{mbs_hit_segment_cnt}个')
    print(f'Average Request Response Time (only successful request): {res_avg_response_time}')

    # baselines
    if vehicle_cache_alg_index != len(p.CACHE_ALGS) - 1 or rsu_cache_alg_index != len(p.CACHE_ALGS) - 1:
        # 将baseline结果输出到结果文件夹
        alg_tested_flag = False
        if Path('./Outcome/baselines/outcome.txt').is_file():
            with open('./Outcome/baselines/outcome.txt', 'r') as file:
                for line in file.readlines():
                    veh_alg_name, rsu_alg_name = (line.split(':')[0]).split('-')
                    if veh_alg_name == p.CACHE_ALGS[vehicle_cache_alg_index] and rsu_alg_name == p.CACHE_ALGS[rsu_cache_alg_index]:
                        alg_tested_flag = True
                        break

        if not alg_tested_flag:
            with open('./Outcome/baselines/outcome.txt', 'a+') as file:
                file.write(f'{p.CACHE_ALGS[vehicle_cache_alg_index]}-{p.CACHE_ALGS[rsu_cache_alg_index]}:{res_succ_request_ratio} {res_cache_hit_ratio} {res_avg_response_time}\n')
        break

    for delegate in overall_cache_delegates_list:
        delegate.save(i_episode=episode_i)

    #test drl cache algs
    if episode_i % args.drl_agent_test_interval == 0 and episode_i != 0:
        print(f'测试第{episode_i}轮模型')
        with open('./Outcome/test/DRL-DRL/seed1/reward.txt','a+') as f:
            f.write('\n'+f'test {episode_i/args.drl_agent_test_interval}'+'\n')
        # reset the environment
        env.reset(test_flag=True)
        while True:
            done = env.step()
            if done:
                break

        # result output
        all_requests_res = env.get_all_request_process_result()
        success_request_cnt = failed_request_cnt = 0
        mbs_hit_segment_cnt = local_cache_hit_segment_cnt = v2v_cache_hit_segment_cnt = v2r_cache_hit_segment_cnt = 0
        total_success_request_handle_duration = 0
        for req in all_requests_res:
            mbs_hit_segment_cnt += req.mbs_hit_segment_cnt
            local_cache_hit_segment_cnt += req.local_cache_hit_segment_cnt
            v2v_cache_hit_segment_cnt += req.v2v_cache_hit_segment_cnt
            v2r_cache_hit_segment_cnt += req.v2r_cache_hit_segment_cnt
            if req.finish_time != -1:
                success_request_cnt += 1
                total_success_request_handle_duration += (req.finish_time - req.origin_time)
            else:
                failed_request_cnt += 1

        for i, metric_i in enumerate(p.METRICS):
            if i == 0:
                metric_res_i = (success_request_cnt / (success_request_cnt + failed_request_cnt)) if success_request_cnt > 0 else 0
            elif i == 1:
                metric_res_i = (total_cache_hit_cnt / (mbs_hit_segment_cnt + total_cache_hit_cnt)) if total_cache_hit_cnt > 0 else 0
            elif i == 2:
                metric_res_i = (total_success_request_handle_duration / success_request_cnt) if success_request_cnt > 0 else 0
            else:
                exit(-1)

            metric_path_i = '-'.join(metric_i.split(' '))
            with open(os.path.join(os.path.join('./Outcome', root_dir_path), f'{metric_path_i}.txt'), 'a+') as f:
                f.write(f'{metric_i}:{metric_res_i}\n')



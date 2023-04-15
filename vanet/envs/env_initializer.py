import os
# import sys
import random
import collections
import vanet.env_params as p
import vanet.tools.models as m
from scipy.stats import poisson


# env_id = sys.argv[1]


def initialize_env(env_id):
    # 生成content library
    sum_jnegbelta = sum(j ** (-p.ZIPF_BELTA) for j in range(1, p.CONTENT_LIBRARY_SIZE + 1))
    zipf_p = [i ** (-p.ZIPF_BELTA) / sum_jnegbelta for i in range(1, p.CONTENT_LIBRARY_SIZE + 1)]   # generate content popularity
    content_library = get_content_library(env_id, zipf_p)

    # 生成MBS和RSU configuration (ID and location)
    nodes_config_dict, center_area = get_rsus_and_mbs_configuration(env_id)

    # 生成vehicle trace和request trace
    vehicle_trace, request_trace, varying_content_popularity_dist, initial_cache_dict, neighbor_vehicles_dict = get_traces_and_randomly_initialize(env_id, content_library, nodes_config_dict, center_area)

    return content_library, vehicle_trace, request_trace, varying_content_popularity_dist, nodes_config_dict, initial_cache_dict, neighbor_vehicles_dict


def get_rsus_and_mbs_configuration(env_id):
    nodes_file = f'./vanet/envs/{env_id}/nodes.txt'  # MBS data in the first row, RSUs data in the following rows
    assert os.path.isfile(nodes_file), 'nonexistent nodes configuration file'

    center_area = {}
    center_nodes_str = ['left_bottom', 'left_up', 'right_bottom', 'right_up']
    nodes_config_dict = {'MBS': None, 'RSU': []}
    rsus_shortest_path_matrix = []

    rsu_cnt = mbs_cnt = 0
    with open(nodes_file, 'r') as file:
        for i, row in enumerate(file.readlines()):
            if i == 0:
                mbs_cnt, rsu_cnt = map(int, row.split(' '))  # 1, 6
            elif 1 <= i <= 4:
                x, y = map(int, row.split(' '))
                center_area[center_nodes_str[i - 1]] = (x, y)
            elif 5 <= i <= 4 + mbs_cnt + rsu_cnt:    # 11
                node_id, node_x, node_y = map(float, row.split(' '))
                if node_id == 0:  # MBS
                    nodes_config_dict['MBS'] = (int(node_id), node_x, node_y)  # 0 ,1000, 1000
                # else:  # RSUs
                #     nodes_config_dict['RSU'].append((int(node_id) - 1, node_x, node_y))
            # else:
            #     rsus_shortest_path_matrix.append(list(map(int, row.split(' '))))

    return nodes_config_dict, center_area


def get_content_library(env_id, zipf_p):
    content_library = {}

    content_library_file = f'./vanet/envs/{env_id}/content_library.txt'
    if not os.path.isfile(content_library_file):
        with open(content_library_file, 'w+') as file:
            for content_i in range(p.CONTENT_LIBRARY_SIZE):
                segment_units = random.randint(p.ENCODED_SEGMENTS_UNITS_MIN, p.ENCODED_SEGMENTS_UNITS_MAX)
                segment_unit_size = random.choice(list(range(p.SEGMENT_UNIT_SIZE_MIN, p.SEGMENT_UNIT_SIZE_MAX + p.SEGMENT_UNIT_SIZE_STEP, p.SEGMENT_UNIT_SIZE_STEP)))
                file.write(f'{content_i} {zipf_p[content_i]} {segment_units} {segment_unit_size}\n')

    with open(content_library_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            content_i, popularity_i, segment_unit_i, segment_unit_size_i = map(float, line.split(' '))
            content_library[content_i] = m.Content(int(content_i), popularity_i, int(segment_unit_i), int(segment_unit_size_i))

    return content_library


def make_sort_trace(raw_trace):
    raw_trace.sort(key=lambda x: (x[3], x[2]))
    
    l, r = 0, 0
    while l < len(raw_trace):
        r = l
        while r < len(raw_trace) - 1 and raw_trace[r + 1][3] == raw_trace[l][3]:
            r += 1
        tmp = [x[0:2] for x in raw_trace[l:r]]
        # print(tmp)
        reverse_flag = False
        if random.randint(1, 100) > 50:
            reverse_flag = True
        tmp.sort(key=lambda x: x[1], reverse=reverse_flag)
        for i in range(0, len(tmp)):
            raw_trace[i + l][0], raw_trace[i+l][1] = tmp[i][0], tmp[i][1]

        cnt = int(random.randrange(200, 300) * (r - l + 1) / 1000)     
        for _ in range(cnt):
            x, y = random.randint(l, r), random.randint(l, r)
            raw_trace[x][0], raw_trace[y][0] = raw_trace[y][0], raw_trace[x][0]
            raw_trace[x][1], raw_trace[y][1] = raw_trace[y][1], raw_trace[x][1]

        l = r + 1
    # tmp.sort(key=lambda x: x[0])
    raw_trace.sort(key=lambda x: x[0])
    return raw_trace


def get_traces_and_randomly_initialize(env_id, content_library, nodes_config_dict, center_area):
    vehicle_data_file = f'./vanet/envs/{env_id}/vehicle_trace.txt'
    request_data_file = f'./vanet/envs/{env_id}/request_trace.txt'
    varying_popularity_data_file = f'./vanet/envs/{env_id}/varying_popularity_trace.txt'
    initial_cache_data_file = f'./vanet/envs/{env_id}/initial_cache_data.txt'
    # rsu_neighbor_veh_file = f'./vanet/envs/{env_id}/rsu_neighbor_vehs.txt'
    veh_neighbor_veh_file = f'./vanet/envs/{env_id}/veh_neighbor_vehs.txt'
    # veh_neighbor_rsu_file = f'./vanet/envs/{env_id}/veh_neighbor_rsus.txt'
    vehicle_trace, request_trace = [], []
    vehicle_trace_dict = {}
    varying_content_popularity_dist = {}
    initial_cache_dict = {'RSU': {}, 'Vehicle': {}}
    neighbor_vehicles_dict = {}

    vehicle_trace_time = None
    max_vehicle_id = -1

    assert os.path.isfile(vehicle_data_file), 'vehicle trace not exists'

    # 读取vehicle trace
    with open(vehicle_data_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'step' in line:
                vehicle_trace_time = int(line.split(' ')[1].split('\n')[0])
                vehicle_trace_dict[vehicle_trace_time] = {}
                continue
            assert vehicle_trace_time is not None, 'invalid time'
            v_id, x, y = map(float, line.split(' '))
            vehicle_trace.append((vehicle_trace_time, int(v_id), x, y))
            vehicle_trace_dict[vehicle_trace_time][int(v_id)] = (x, y)
            max_vehicle_id = max(max_vehicle_id, int(v_id))

    # assert vehicle_trace_time > 300, 'too short trace'

    request_trace_origin_time, request_trace_last_time = vehicle_trace_time // 3, vehicle_trace_time * 2 // 3
    # request_trace_origin_time, request_trace_last_time = 0, vehicle_trace_time
    vehicle_num = max_vehicle_id + 1    # vehicle总数

    if not os.path.isfile(request_data_file):   # 如果request trace不存在，则根据vehicle trace生成request trace并写入文件

        tmp_neighbor_vehicles_dict, tmp_neighbor_rsus_dict, tmp_rsu_neighbor_vehicles_dict = get_neighbors(vehicle_trace_dict, nodes_config_dict)

        # with open(rsu_neighbor_veh_file, 'w+') as file:
        #     for t_i in list(vehicle_trace_dict.keys()):
        #         file.write(f'Step-{t_i}\n')
        #
        #         for rsu_id, rsu_x, rsu_y in nodes_config_dict['RSU']:
        #             if len(tmp_rsu_neighbor_vehicles_dict[t_i][rsu_id]) == 0:
        #                 continue
        #             file.write(f'{rsu_id}:')
        #             for veh_id in tmp_rsu_neighbor_vehicles_dict[t_i][rsu_id]:
        #                 file.write(f'{veh_id} ')
        #             file.write('\n')

        with open(veh_neighbor_veh_file, 'w+') as file:
            for t_i in list(vehicle_trace_dict.keys()):
                file.write(f'Step-{t_i}\n')

                for veh_id in vehicle_trace_dict[t_i].keys():
                    if len(tmp_neighbor_vehicles_dict[t_i][veh_id]) == 0:
                        continue
                    file.write(f'{veh_id}:')
                    for neigh_veh_id in tmp_neighbor_vehicles_dict[t_i][veh_id]:
                        file.write(f'{neigh_veh_id} ')
                    file.write('\n')

        # with open(veh_neighbor_rsu_file, 'w+') as file:
        #     for t_i in list(vehicle_trace_dict.keys()):
        #         file.write(f'Step-{t_i}\n')
        #
        #         for veh_id in vehicle_trace_dict[t_i].keys():
        #             if len(tmp_neighbor_rsus_dict[t_i][veh_id]) == 0:
        #                 continue
        #             file.write(f'{veh_id}:')
        #             for neigh_rsu_id in tmp_neighbor_rsus_dict[t_i][veh_id]:
        #                 file.write(f'{neigh_rsu_id} ')
        #             file.write('\n')

        # 生成所有vehicle的preference distribution以及request probability
        dists, prior_veh_probs = get_vehicle_preference_distributions(vehicle_num, content_library)

        # 生成每个时隙的content popularity分布
        tmp_content_popularity_dist = produce_content_popularity_distribution_in_each_timeslot(vehicle_trace_time,
                                                                                               vehicle_trace_dict,
                                                                                               prior_veh_probs,
                                                                                               dists,
                                                                                               center_area)

        # 每个时隙的content popularity分布写入文件
        with open(varying_popularity_data_file, 'w+') as file:
            for t_i in list(tmp_content_popularity_dist.keys()):
                file.write(f'{t_i} ')
                for content_i in range(p.CONTENT_LIBRARY_SIZE):
                    file.write(f'{tmp_content_popularity_dist[t_i][content_i]} ')
                file.write('\n')

        # 生成request trace
        tmp_request_trace = produce_request_trace(request_trace_origin_time,
                                                  request_trace_last_time,
                                                  vehicle_trace_dict,
                                                  prior_veh_probs, dists,
                                                  content_library,
                                                  center_area)

        # request trace写入文件
        with open(request_data_file, 'w+') as file:
            raw_r_trace = []
            for t_i in list(tmp_request_trace.keys()):
                for r_id, origin_time, content_id, vehicle_id, segment_units, segment_unit_size in tmp_request_trace[t_i]:
                    file.write(f'{r_id} {origin_time} {content_id} {vehicle_id} {segment_units} {segment_unit_size}\n')
                    raw_r_trace.append([r_id, origin_time, content_id, vehicle_id, segment_units, segment_unit_size])
            # likely_sorted_trace = make_sort_trace(raw_r_trace)      
            # for r in likely_sorted_trace:
                # print(*r, sep=f' ', file=file)

        # 随机初始化所有边缘节点的缓存空间
        tmp_initial_cache_dict = randomly_initialize_D2D_cache_space(vehicle_num,
                                                                     nodes_config_dict,
                                                                     dists,
                                                                     content_library)

        # initial cache写入文件 (先RSU缓存, 后Vehicle缓存)
        with open(initial_cache_data_file, 'w+') as file:
            for rsu_i in tmp_initial_cache_dict['RSU']:
                file.write(f'RSU {rsu_i}: ')
                for con_id, seg_id in tmp_initial_cache_dict['RSU'][rsu_i]:
                    file.write(f'{con_id} {seg_id} ')
                file.write('\n')
            for veh_i in tmp_initial_cache_dict['Vehicle']:
                file.write(f'Vehicle {veh_i}: ')
                for con_id, seg_id in tmp_initial_cache_dict['Vehicle'][veh_i]:
                    file.write(f'{con_id} {seg_id} ')
                file.write('\n')

    assert os.path.isfile(request_data_file), 'request trace not exists'
    assert os.path.isfile(varying_popularity_data_file), 'varying popularity content trace not exists'
    assert os.path.isfile(initial_cache_data_file), 'initial cache data not exists'

    # 读取每个时隙的content popularity分布
    with open(varying_popularity_data_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line_data_list = list(map(float, line.split(' ')[:-1]))
            t_i, content_popularity_dist_i = int(line_data_list[0]), line_data_list[1:]
            varying_content_popularity_dist[t_i] = content_popularity_dist_i
    varying_content_popularity_dist[0] = [content_library[content_i].popularity for content_i in content_library.keys()]
    
    # 读取request trace
    with open(request_data_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            request_id, origin_time, content_id, vehicle_id, segment_units, segment_unit_size = map(int, line.split(' '))
            request_trace.append((request_id, origin_time, content_id, vehicle_id, segment_units, segment_unit_size))

    # 读取initial cache
    with open(initial_cache_data_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            node_str, cache_config_str = list(line.split(': '))
            node_type, node_id = node_str.split(' ')[0], int(node_str.split(' ')[1])
            assert node_type in initial_cache_dict, 'invalid node type for initial_cache_dict'
            assert node_id not in initial_cache_dict[node_type], 'invalid node_id for initial_cache_dict'

            initial_cache_dict[node_type][node_id] = []
            cache_config_list = list(cache_config_str.split(' '))[:-1]
            assert len(cache_config_list) % 2 == 0, 'Wrong cache_config_list'
            for cache_id in range(len(cache_config_list) // 2):
                content_x, seg_x = cache_config_list[cache_id * 2], cache_config_list[cache_id * 2 + 1]
                initial_cache_dict[node_type][node_id].append((int(content_x), int(seg_x)))

    # with open(rsu_neighbor_veh_file, 'r') as file:
    #     time_step = None
    #     for line in file.readlines():
    #         if 'Step' in line:
    #             time_step = int(line.split('-')[1])
    #             rsu_neighbor_vehicles_dict[time_step] = collections.defaultdict(list)
    #             continue
    #         rsu_id, adj_veh_id_list = int(line.split(':')[0]), list(map(int, (line.split(':')[1]).split(' ')[:-1]))
    #         for adj_veh_id in adj_veh_id_list:
    #             rsu_neighbor_vehicles_dict[time_step][rsu_id].append(adj_veh_id)

    with open(veh_neighbor_veh_file, 'r') as file:
        time_step = None
        for line in file.readlines():
            if 'Step' in line:
                time_step = int(line.split('-')[1])
                neighbor_vehicles_dict[time_step] = collections.defaultdict(list)
                continue
            veh_id, adj_veh_id_list = int(line.split(':')[0]), list(map(int, (line.split(':')[1]).split(' ')[:-1]))
            for adj_veh_id in adj_veh_id_list:
                neighbor_vehicles_dict[time_step][veh_id].append(adj_veh_id)

    # with open(veh_neighbor_rsu_file, 'r') as file:
    #     time_step = None
    #     for line in file.readlines():
    #         if 'Step' in line:
    #             time_step = int(line.split('-')[1])
    #             neighbor_rsus_dict[time_step] = collections.defaultdict(list)
    #             continue
    #         veh_id, adj_rsu_id_list = int(line.split(':')[0]), list(map(int, (line.split(':')[1]).split(' ')[:-1]))
    #         for adj_rsu_id in adj_rsu_id_list:
    #             neighbor_rsus_dict[time_step][veh_id].append(adj_rsu_id)

    return vehicle_trace, request_trace, varying_content_popularity_dist, initial_cache_dict, neighbor_vehicles_dict


def produce_content_popularity_distribution_in_each_timeslot(vehicle_trace_time, vehicle_trace_dict, prior_veh_probs, dists, center_area):
    tmp_content_popularity_dist = {}
    for t_i in range(vehicle_trace_time + 1):
        if t_i in vehicle_trace_dict and len(vehicle_trace_dict[t_i].keys()) > 0:
            tmp_content_popularity_dist[t_i] = []

            existing_vehicles_ids = [key for key in vehicle_trace_dict[t_i].keys() if
                                     center_area['left_bottom'][0] <= vehicle_trace_dict[t_i][key][0] <=
                                     center_area['right_up'][0] and
                                     center_area['left_bottom'][1] <= vehicle_trace_dict[t_i][key][1] <=
                                     center_area['right_up'][1]]
            total_probs = sum(prior_veh_probs[veh_id] for veh_id in existing_vehicles_ids)
            cur_prior_veh_probs = {}
            for veh_id in existing_vehicles_ids:
                cur_prior_veh_probs[veh_id] = prior_veh_probs[veh_id] / total_probs
            # assert sum(cur_prior_veh_probs) == 1, 'wrong cur_prior_veh_probs implementation'

            for content_i in range(p.CONTENT_LIBRARY_SIZE):
                pr_i = sum(cur_prior_veh_probs[veh_id] * dists[veh_id][content_i] for veh_id in existing_vehicles_ids)
                tmp_content_popularity_dist[t_i].append(pr_i)
            # assert sum(tmp_content_popularity_dist[t_i]) == 1, 'wrong tmp_content_popularity_dist implementation'

    return tmp_content_popularity_dist


def produce_request_trace(request_trace_origin_time, request_trace_last_time, vehicle_trace_dict, prior_veh_probs, dists, content_library, center_area):
    tmp_request_trace = {}
    req_id = -1
    for t_i in range(request_trace_origin_time, request_trace_last_time + 1):
        tmp_request_trace[t_i] = []
        req_cnt = poisson.rvs(mu=p.REQUEST_DENSE, size=1)[0]
        existing_vehicles_ids = [key for key in vehicle_trace_dict[t_i].keys() if
                                 center_area['left_bottom'][0] <= vehicle_trace_dict[t_i][key][0] <=
                                 center_area['right_up'][0] and
                                 center_area['left_bottom'][1] <= vehicle_trace_dict[t_i][key][1] <=
                                 center_area['right_up'][1]]
        if not existing_vehicles_ids:
            continue
        for req_i in range(req_cnt):
            req_id += 1
            # veh_id = random.choice(existing_vehicles_ids)   # Setting1: 区域中的车辆等概率发起content request
            veh_id = random.choices(existing_vehicles_ids, weights=[prior_veh_probs[x] for x in existing_vehicles_ids], k=1)[0]  # Setting2: 区域中的车辆基于先验概率分布发起content request
            req_content_id = random.choices(list(range(p.CONTENT_LIBRARY_SIZE)), weights=dists[veh_id], k=1)[0]
            tmp_request_trace[t_i].append(
                (req_id, t_i, req_content_id, veh_id, content_library[req_content_id].segment_units,
                 content_library[req_content_id].segment_unit_size))
    return tmp_request_trace


def randomly_initialize_D2D_cache_space(vehicle_num, nodes_config_dict, dists, content_library):
    initial_cache_dict = {'RSU': {}, 'Vehicle': {}}

    # 随机生成RSUs上的缓存
    content_popularity_distribution = [content_library[content_i].popularity for content_i in content_library.keys()]
    for rsu_i in range(len(nodes_config_dict['RSU'])):
        initial_cache_dict['RSU'][rsu_i] = set()
        total_cache_size = 0
        while total_cache_size < p.RSU_CACHE_SIZE - p.SEGMENT_UNIT_SIZE_MAX:
            cache_content_id = random.randint(0, p.CONTENT_LIBRARY_SIZE - 1)
            seg_id = random.randint(0, content_library[cache_content_id].segment_units - 1)
            if (cache_content_id, seg_id) not in initial_cache_dict['RSU'][rsu_i] and total_cache_size + content_library[cache_content_id].segment_unit_size <= p.RSU_CACHE_SIZE:
                initial_cache_dict['RSU'][rsu_i].add((cache_content_id, seg_id))
                total_cache_size += content_library[cache_content_id].segment_unit_size
            else:
                continue

    # 随机生成Vehicles上的缓存
    for veh_i in range(vehicle_num):
        initial_cache_dict['Vehicle'][veh_i] = set()
        total_cache_size = 0
        while total_cache_size < p.VEH_CACHE_SIZE - p.SEGMENT_UNIT_SIZE_MAX:
            cache_content_id = random.randint(0, p.CONTENT_LIBRARY_SIZE - 1)
            seg_id = random.randint(0, content_library[cache_content_id].segment_units - 1)
            if (cache_content_id, seg_id) not in initial_cache_dict['Vehicle'][veh_i] and total_cache_size + content_library[cache_content_id].segment_unit_size <= p.VEH_CACHE_SIZE:
                initial_cache_dict['Vehicle'][veh_i].add((cache_content_id, seg_id))
                total_cache_size += content_library[cache_content_id].segment_unit_size
            else:
                continue

    return initial_cache_dict


def get_neighbors(vehicle_trace_dict, nodes_config_dict):
    def verify_if_neighbor_rsu(node_x, node_y, rsu_j_x, rsu_j_y):
        return (node_x - rsu_j_x) ** 2 + (node_y - rsu_j_y) ** 2 <= p.RSU_RANGE ** 2

    def verify_if_neighbor_vehicle(node_x, node_y, veh_j_x, veh_j_y):
        return (node_x - veh_j_x) ** 2 + (node_y - veh_j_y) ** 2 <= p.VEH_RANGE ** 2

    tmp_neighbor_vehicles_dict, tmp_neighbor_rsus_dict, tmp_rsu_neighbor_vehicles_dict = {}, {}, {}
    for t_j in vehicle_trace_dict.keys():
        tmp_neighbor_vehicles_dict[t_j] = collections.defaultdict(list)
        tmp_neighbor_rsus_dict[t_j] = collections.defaultdict(list)
        tmp_rsu_neighbor_vehicles_dict[t_j] = collections.defaultdict(list)

        for veh_id in vehicle_trace_dict[t_j].keys():
            veh_x, veh_y = vehicle_trace_dict[t_j][veh_id][0], vehicle_trace_dict[t_j][veh_id][1]

            for rsu_id, rsu_x, rsu_y in nodes_config_dict['RSU']:
                if verify_if_neighbor_rsu(veh_x, veh_y, rsu_x, rsu_y):
                    if rsu_id not in tmp_neighbor_rsus_dict[t_j][veh_id]:
                        tmp_neighbor_rsus_dict[t_j][veh_id].append(rsu_id)
                    if veh_id not in tmp_rsu_neighbor_vehicles_dict[t_j][rsu_id]:
                        tmp_rsu_neighbor_vehicles_dict[t_j][rsu_id].append(veh_id)

            for veh_j_id in vehicle_trace_dict[t_j].keys():
                if veh_id == veh_j_id:
                    continue

                if verify_if_neighbor_vehicle(veh_x, veh_y, vehicle_trace_dict[t_j][veh_j_id][0],
                                              vehicle_trace_dict[t_j][veh_j_id][1]):
                    if veh_j_id not in tmp_neighbor_vehicles_dict[t_j][veh_id]:
                        tmp_neighbor_vehicles_dict[t_j][veh_id].append(veh_j_id)
                    if veh_id not in tmp_neighbor_vehicles_dict[t_j][veh_j_id]:
                        tmp_neighbor_vehicles_dict[t_j][veh_j_id].append(veh_id)

            tmp_neighbor_vehicles_dict[t_j][veh_id].sort(key=lambda x: (vehicle_trace_dict[t_j][x][0] - veh_x) ** 2 + (
                        vehicle_trace_dict[t_j][x][1] - veh_y) ** 2)
            tmp_neighbor_rsus_dict[t_j][veh_id].sort(key=lambda x: (nodes_config_dict['RSU'][x][1] - veh_x) ** 2 + (
                        nodes_config_dict['RSU'][x][2] - veh_y) ** 2)

    return tmp_neighbor_vehicles_dict, tmp_neighbor_rsus_dict, tmp_rsu_neighbor_vehicles_dict


def get_vehicle_preference_distributions(vehicle_num, content_library):
    # kernel function g(x_k, y_f): correlation between vehicle k and content f
    g = [[0] * p.CONTENT_LIBRARY_SIZE for _ in range(vehicle_num)]
    for k in range(vehicle_num):
        for f in range(p.CONTENT_LIBRARY_SIZE):
            x_k, y_f = random.uniform(0, 1), random.uniform(0, 1)
            g[k][f] = (1 - abs(x_k - y_f)) ** (1 / p.KERNEL_ALPHA ** 3 - 1)

    sum_g = [sum(g[k][f] for k in range(vehicle_num)) for f in range(p.CONTENT_LIBRARY_SIZE)]
    joint_probs = [[0] * p.CONTENT_LIBRARY_SIZE for _ in range(vehicle_num)]
    for k in range(vehicle_num):
        for f in range(p.CONTENT_LIBRARY_SIZE):
            joint_probs[k][f] = content_library[f].popularity * g[k][f] / sum_g[f]

    prior_veh_probs = [sum(joint_probs[x]) for x in range(vehicle_num)]
    dists = [[0] * p.CONTENT_LIBRARY_SIZE for _ in range(vehicle_num)]
    for k in range(vehicle_num):
        for f in range(p.CONTENT_LIBRARY_SIZE):
            dists[k][f] = joint_probs[k][f] / prior_veh_probs[k]

    return dists, prior_veh_probs
'''
    pref = open("user_preference.txt", 'w+')
    for k in range(vehicle_num):
        for f in range(p.CONTENT_LIBRARY_SIZE):
            print(round(dists[k][f], 2), end='\t', file=pref)
        print(file=pref)
    pref.close()
'''


# 读取.csv文件存储的content requests
# def get_vehicle_preference_distribution(env_id, v_cnt):
#     csv_file = f'./{env_id}/v_preference.csv'
#     with open(csv_file) as f:
#         reader = csv.reader(f)
#         header_row = next(reader)
#         for row in reader:
#             print(row)
#             v_id, c_id, rating, _ = [int(float(x)) for x in row]
#             if v_id > v_cnt:
#                 break


# test
# initialize_env('berlin')

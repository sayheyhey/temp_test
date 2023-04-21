import os
import math
import logging
import collections
import numpy as np
import vanet.env_params as p
import vanet.tools.models as m
import vanet.envs.env_initializer as e


class Env:
    def __init__(self, env_id, cache_delegates_list, rsu_cache_enabled, veh_cache_enabled):
        self.env_id = env_id # berlin
        self.cache_delegates_list = cache_delegates_list
        self.rsu_cache_enabled, self.veh_cache_enabled = rsu_cache_enabled, veh_cache_enabled
        self.content_library, self.v_trace, self.r_trace, self.varying_content_popularity_dist, self.nodes_config_dict, self.initial_cache_dict, self.neighbor_vehicles_dict = e.initialize_env(
            env_id)

        self.last_valid_time = self.v_trace[-1][0]
        self.timer = 0  # 系统时钟
        self.processing_requests = None  # 存储所有未完成处理的requests的信息 (dict)
        self.processing_rsu_requests = None  # 存储所有未完成处理的rsu requests的信息 (dict)
        self.all_vehicle_data = None  # 存储所有vehicles的信息 (dict)
        self.all_request_data = None  # 存储所有requests的信息 (list)
        self.vehicle_trace, self.request_trace = None, None
        self.test_flag = False

        # configure RSUs and MBS
        # self.rsus = {}
        self.mbs = m.MBS(self.nodes_config_dict['MBS'][0], self.nodes_config_dict['MBS'][1],
                         self.nodes_config_dict['MBS'][2])
        # self.sorted_list = defaultdict(list)

        self.reset()

    def reset(self, test_flag=False):
        self.test_flag = test_flag

        # configure RSUs
        # self.rsus = {}
        # for rsu_i in range(len(self.nodes_config_dict['RSU'])):
        #     node_id, node_x, node_y = self.nodes_config_dict['RSU'][rsu_i][0], self.nodes_config_dict['RSU'][rsu_i][1], \
        #                               self.nodes_config_dict['RSU'][rsu_i][2]
        #     assert rsu_i == node_id, 'rsu_i not equals to node_id'
        #
        #     self.rsus[node_id] = m.RSU(node_id, node_x, node_y,
        #                                cache_delegate=self.cache_delegates_list[1],
        #                                s=p.RSU_CACHE_SIZE)
        #     # 初始化RSU缓存空间
        #     if node_id in self.initial_cache_dict['RSU'] and self.rsu_cache_enabled:
        #         for content_x, seg_x in self.initial_cache_dict['RSU'][node_id]:
        #             self.update_cache_space_on_node(self.rsus[node_id], content_x, seg_x, 0)

        self.vehicle_trace, self.request_trace = {}, {}

        # 生成vehicle trace字典
        for t_i, veh_id, veh_x, veh_y in self.v_trace:
            if t_i not in self.vehicle_trace:
                self.vehicle_trace[t_i] = {}
            self.vehicle_trace[t_i][veh_id] = (veh_id, veh_x, veh_y)

        # 生成content trace字典
        for request_id, origin_time, content_id, vehicle_id, segment_units, segment_unit_size in self.r_trace:
            if origin_time not in self.request_trace:
                self.request_trace[origin_time] = []
            self.request_trace[origin_time].append(
                m.Request(request_id, origin_time, content_id, vehicle_id, segment_units, segment_unit_size))

        self.timer = 0
        self.processing_requests = {}
        self.processing_rsu_requests = collections.defaultdict(list)
        self.all_vehicle_data = {}
        self.all_request_data = []

    def step(self, cache_enabled=True):
        # 更新系统时钟
        self.timer += 1
        if self.timer % 100 == 0:
            print(f'时隙{self.timer}开始')

        cur_vehicles = {}  # 存储timer时刻环境中的vehicles信息

        assert 0 <= self.timer <= self.last_valid_time, f'timer exceeds valid simulation trace'

        if self.timer in self.vehicle_trace:
            # 更新self.timer时刻环境中的vehicles
            for v_id in self.vehicle_trace[self.timer].keys():
                vehicle_i, vehicle_x, vehicle_y = self.vehicle_trace[self.timer][v_id]
                if v_id not in self.all_vehicle_data:  # 新出现的vehicle
                    self.all_vehicle_data[v_id] = m.Vehicle(vehicle_i, vehicle_x, vehicle_y,
                                                            cache_delegate=self.cache_delegates_list[0],
                                                            s=p.VEH_CACHE_SIZE)
                    # 初始化Vehicle缓存空间
                    # if vehicle_i in self.initial_cache_dict['Vehicle'] and self.veh_cache_enabled:
                    #     for content_x, seg_x in self.initial_cache_dict['Vehicle'][vehicle_i]:
                    #         self.update_cache_space_on_node(self.all_vehicle_data[v_id], content_x, seg_x, self.timer)

                self.all_vehicle_data[v_id].update_location(vehicle_x, vehicle_y)
                self.all_vehicle_data[v_id].refresh_in_comm_time_per_slot()
                cur_vehicles[v_id] = self.all_vehicle_data[v_id]

            # 更新区域中所有vehicles的邻居vehicles和邻居RSU
            # self.get_neighbours_veh(cur_vehicles)
            for v_id in cur_vehicles.keys():
                neighbor_vehicles = self.get_adjacent_vehicles_for_veh(v_id)
                # neighbor_rsus = self.get_adjacent_rsus(v_id)
                self.all_vehicle_data[v_id].update_adjacent_vehicles(neighbor_vehicles)
                # self.all_vehicle_data[v_id].update_adjacent_rsus(neighbor_rsus)

            # for rsu_id in self.rsus.keys():
            #     neighbor_vehicles = self.get_adjacent_vehicles_for_rsu(rsu_id)
            #     self.rsus[rsu_id].update_adjacent_vehicles(neighbor_vehicles)

        # 丢弃timer时刻已失效的request (发起request的车辆已出界)
        for req_id in list(self.processing_requests.keys()):
            if self.processing_requests[req_id].vehicle_id not in cur_vehicles:
                logging.debug(f'Request{req_id}: content{self.processing_requests[req_id].content_id}失效')
                # 如果req_id的target node是vehicle，需要重置target node的通信状态
                target_node = self.processing_requests[req_id].last_timeslot_segment_provider
                if target_node and target_node.type == 'Vehicle':
                    target_node.in_comm_flag = False
                del self.processing_requests[req_id]

        if self.timer in self.request_trace:
            # 更新timer时刻环境中新产生的request
            new_requests = self.request_trace[self.timer]
            for req in new_requests:
                self.all_request_data.append(req)

                # 更新发起车辆上的local request cnt，邻居node上的remote request cnt
                cur_vehicles[req.vehicle_id].increment_local_request_cnt(self.timer, req.content_id)
                for neighbor_v in cur_vehicles[req.vehicle_id].neighbor_vehicles:
                    neighbor_v.increment_remote_request_cnt(self.timer, req.content_id)
                # for neighbor_r in cur_vehicles[req.vehicle_id].neighbor_rsus:
                #     neighbor_r.increment_remote_request_cnt(self.timer, req.content_id)

                request_id = req.r_id
                self.processing_requests[request_id] = req
                cur_vehicles[req.vehicle_id].cur_request_cnt += 1

        # # 更新RSU上新收到的content segment
        # for rsu_request_i in self.processing_rsu_requests[self.timer]:
        #     self.update_cache_space_on_node(self.rsus[rsu_request_i.rsu_id], rsu_request_i.content_id,
        #                                     rsu_request_i.seg_id, self.timer)

        # 处理所有未完成requests
        self.deliver_contents(cur_vehicles, cache_enabled=cache_enabled)

        return self.timer == self.last_valid_time

    def deliver_contents(self, cur_vehicles, cache_enabled):
        while True:
            all_complete_flag = True  # 该时隙的所有request的处理逻辑是否已完成

            for req_id in list(self.processing_requests.keys()):
                req = self.processing_requests[req_id]
                source_vehicle = cur_vehicles[req.vehicle_id]
                source_vehicle_in_comm_time = source_vehicle.in_comm_time_per_slot
                if source_vehicle_in_comm_time >= 1:
                    assert source_vehicle_in_comm_time == 1, 'Invalid source_vehicle_in_comm_time'
                    continue

                all_complete_flag = False  # 未完成的req_id在该时隙内存在继续传输的空余时间

                # 从本地缓存处直接获得content segments
                if self.veh_cache_enabled:
                    for seg_i in range(len(req.left_segment_list)):
                        if req.left_segment_list[seg_i] > 0 and req.content_id in self.all_vehicle_data[
                            req.vehicle_id].cache_content_segments_set and seg_i in \
                                self.all_vehicle_data[req.vehicle_id].cache_content_segments_set[req.content_id]:
                            req.left_segment_list[seg_i] = 0
                            req.local_cache_hit_segment_cnt += 1
                            source_vehicle.increment_cache_hit_size_and_cnt(self.timer, req.content_id, self.content_library[req.content_id].segment_unit_size)
                            source_vehicle.update_cache_properties(content_id=req.content_id,
                                                                   seg_id=seg_i,
                                                                   seg_size=self.content_library[
                                                                       req.content_id].segment_unit_size,
                                                                   nowTime=self.timer + source_vehicle.in_comm_time_per_slot,
                                                                   popularity=
                                                                   self.varying_content_popularity_dist[self.timer][
                                                                       req.content_id])
                        else:
                            # 本地缓存无法为req提供完整的服务
                            source_vehicle.increment_unable_request_cnt(self.timer, req.content_id)
                    if sum(req.left_segment_list) == 0:  # 本地缓存了request content的所有分片
                        # req处理完成
                        logging.debug(
                            f'Request{req.r_id}: Vehicle节点{req.vehicle_id}上本地缓存完成content{req.content_id}的所有分片处理')
                        req.finish_time = self.timer + source_vehicle_in_comm_time
                        cur_vehicles[req.vehicle_id].cur_request_cnt -= 1
                        del self.processing_requests[req_id]
                        continue

                # 从其他Node处获得content segments:
                #   若segment未确定从哪个节点传输，则根据Three steps
                #       (adjacent vehicle nodes -> adjacent rsu nodes -> mbs) 确定由哪个节点提供segment
                #   若确定了segment传输的通信双方，则保持通信至完成segment传输或出现departure loss
                #       考虑传输过程中的departure loss：如果segment的传输过程中通信双方离开通信范围，则该segment传输失败

                # 本时隙source_vehicle可用于传输req的最大时间 \in (0,1]
                # 需要考虑当前时隙剩余时间、source vehicle当前时隙空余时间
                left_time = 1 - source_vehicle_in_comm_time
                assert 0 < left_time <= 1, f'invalid left time for vehicle {source_vehicle.id}'

                if req.last_timeslot_segment_provider:  # 已确定req的服务提供节点
                    # 进入这个分支的唯一情况是: source vehicle和target node在上一时隙有未完成的segment delivery通信
                    target_node = req.last_timeslot_segment_provider

                    assert left_time == 1, 'in_comm_time_per_slot of the source vehicle should be 0 for the continous content delivery'
                    if target_node.type == 'Vehicle':
                        # 上一时隙的target node如果是vehicle，需要考虑当前时隙target node离开区域的情况
                        assert target_node.in_comm_time_per_slot == 0 or target_node.id not in cur_vehicles, 'unexpected situation for the continous content delivery'
                        if target_node.in_comm_time_per_slot != 0:  # target node离开区域
                            assert target_node.id not in cur_vehicles, 'unexpected situation 1 for the continous content delivery'
                        else:
                            assert target_node.id in cur_vehicles, 'unexpected situation 2 for the continous content delivery'

                    # 确定正在传输的segment
                    target_seg = None
                    for seg_i in range(len(req.left_segment_list)):
                        if 0 < req.left_segment_list[seg_i] < self.content_library[req.content_id].segment_unit_size:
                            target_seg = seg_i
                            break
                    assert target_seg is not None, 'no transmitting segment'

                    # 继续传输req
                    departure_flag, d_t = self.continue_content_segment_delivery(req, cur_vehicles, target_seg,
                                                                                 left_time=left_time)

                    if departure_flag:  # departure loss -> 需要重新确定req的服务提供节点
                        d_t, target_node = self.initiate_new_content_request_delivery(req, cur_vehicles,
                                                                                      cache_enabled=cache_enabled,
                                                                                      left_time=left_time)
                    else:  # 无departure loss -> 2种情况：1.完成target_seg的传输; 2.当前时隙无法完成target_seg的传输
                        assert req.left_segment_list[
                                   target_seg] == 0 or d_t == 1, 'unexpected situation for continous segment delivery'

                    # 更新continuous content delivery时通信双方节点的当前时隙空闲时间
                    source_vehicle.in_comm_time_per_slot = min(1, source_vehicle.in_comm_time_per_slot + d_t)
                    if target_node.type == 'Vehicle':
                        target_node.in_comm_time_per_slot = min(1, target_node.in_comm_time_per_slot + d_t)
                else:  # 未确定req的服务提供节点
                    if not source_vehicle.in_comm_flag:  # 未处于通信状态的节点才能发起新的content request
                        d_t, target_node = self.initiate_new_content_request_delivery(req, cur_vehicles,
                                                                                      cache_enabled=cache_enabled,
                                                                                      left_time=left_time)
                        # 更新new content delivery时通信双方节点的当前时隙空闲时间
                        source_vehicle.in_comm_time_per_slot = min(1, source_vehicle.in_comm_time_per_slot + d_t)
                        if target_node.type == 'Vehicle':
                            target_node.in_comm_time_per_slot = min(1, target_node.in_comm_time_per_slot + d_t)
                    else:  # source vehicle处于通信状态，content request阻塞
                        continue

                if sum(req.left_segment_list) == 0:
                    # req处理完成
                    logging.debug(f'Request{req.r_id}: content{req.content_id}的所有分片完成处理')
                    req.finish_time = self.timer + source_vehicle.in_comm_time_per_slot
                    cur_vehicles[req.vehicle_id].cur_request_cnt -= 1
                    del self.processing_requests[req_id]
                    continue

            if all_complete_flag:
                break

    def continue_content_segment_delivery(self, request, cur_vehicles, target_seg, left_time):
        assert request.last_timeslot_segment_provider, f'No invalid last_timeslot_segment_provider for request {request.r_id}'
        assert left_time > 0, 'Invalid time for continuous segment delivery'
        # assert request.content_id in request.last_timeslot_segment_provider.cache_content_segments_set and target_seg in \
        #        request.last_timeslot_segment_provider.cache_content_segments_set[
        #            request.content_id], 'assert seg replaced'

        duration = left_time
        departure_flag = False
        source_vehicle = self.get_vehicle_from_vid(cur_vehicles, request.vehicle_id)

        assert 0 <= source_vehicle.in_comm_time_per_slot < 1, 'source vehicle has no free time in continuous segment delivery'
        assert source_vehicle.in_comm_flag, 'source vehicle not in communication in continuous segment delivery'

        commu_dist = self.get_distance(source_vehicle, request.last_timeslot_segment_provider)
        provider_type = request.last_timeslot_segment_provider.type

        # departure loss
        if (provider_type == 'RSU' and commu_dist > p.RSU_RANGE) or (
                provider_type == 'Vehicle' and commu_dist > p.VEH_RANGE) or (
                provider_type == 'Vehicle' and request.last_timeslot_segment_provider.id not in cur_vehicles) or not (
                request.content_id in request.last_timeslot_segment_provider.cache_content_segments_set and target_seg in
                request.last_timeslot_segment_provider.cache_content_segments_set[request.content_id]):

            logging.debug(
                f'Request{request.r_id}: request发起者Vehicle节点{request.vehicle_id}与服务提供者{provider_type}节点{request.last_timeslot_segment_provider.id}断开连接，content{request.content_id}的segment{target_seg}传输失败')
            departure_flag = True
            request.left_segment_list[target_seg] = self.content_library[
                request.content_id].segment_unit_size  # Departure loss: 传输失败，重置segment需求
            if provider_type == 'Vehicle':
                request.last_timeslot_segment_provider.in_comm_flag = False
            request.last_timeslot_segment_provider = None
            source_vehicle.in_comm_flag = False
        else:
            duration = self.transfer_segment_of_request_to_node(source_vehicle, request.last_timeslot_segment_provider,
                                                                request, target_seg, left_time)

        assert 0 < duration, 'Wrong continue_content_segment_delivery func output'
        return departure_flag, min(left_time, duration)

    def initiate_new_content_request_delivery(self, request, cur_vehicles, cache_enabled, left_time):
        assert not request.last_timeslot_segment_provider, f'existing last_timeslot_segment_provider for request {request.r_id}'
        assert left_time > 0, 'Invalid time for new segment delivery'

        duration = left_time
        source_vehicle = self.get_vehicle_from_vid(cur_vehicles, request.vehicle_id)
        target_node, target_seg = None, None
        assert not source_vehicle.in_comm_flag, 'source vehicle in communication in new content delivery'
        assert left_time + source_vehicle.in_comm_time_per_slot == 1, 'Wrong implenmentation'
        assert source_vehicle.in_comm_time_per_slot < 1, 'source_vehicle.in_comm_time_per_slot >= 1'

        process_flag = False
        if cache_enabled:
            # neighbor_vehicles = self.get_adjacent_vehicles_for_veh(source_vehicle, cur_vehicles)
            # neighbor_rsus = self.get_adjacent_rsus(source_vehicle)
            # source_vehicle.update_adjacent_vehicles(neighbor_vehicles)
            # source_vehicle.update_adjacent_rsus(neighbor_rsus)
            neighbor_vehicles = source_vehicle.neighbor_vehicles
            if neighbor_vehicles and self.veh_cache_enabled:  # 从当前无request的邻居Vehicle处请求content_id的内容
                for v_j in neighbor_vehicles:
                    v_j.increment_unable_request_cnt(self.timer,
                                                     request.content_id,
                                                     cnt=self.content_library[request.content_id].segment_units - (len(
                                                         v_j.cache_content_segments_set[
                                                             request.content_id]) if request.content_id in v_j.cache_content_segments_set else 0))
                    if not v_j.in_comm_flag and v_j.in_comm_time_per_slot <= source_vehicle.in_comm_time_per_slot and request.content_id in v_j.cache_content_segments_set and v_j.cur_request_cnt <= 0:
                        assert v_j.cur_request_cnt == 0, 'v_j.cur_request_cnt < 0'
                        for seg in v_j.cache_content_segments_set[request.content_id]:
                            if len(request.left_segment_list)<(seg+2):
                                if request.left_segment_list[seg] > 0 and not process_flag:
                                    process_flag = True
                                    target_node, target_seg = v_j, seg
                                    v_j.in_comm_flag = True
                                    source_vehicle.in_comm_flag = True
                                    break

            # if not process_flag and neighbor_rsus and self.rsu_cache_enabled:  # 再从邻居RSU集合中请求content_id的内容
            #     for rsu_j in neighbor_rsus:
            #         rsu_j.increment_unable_request_cnt(self.timer,
            #                                            request.content_id,
            #                                            cnt=self.content_library[request.content_id].segment_units - (
            #                                                len(rsu_j.cache_content_segments_set[
            #                                                        request.content_id]) if request.content_id in rsu_j.cache_content_segments_set else 0))
            #
            #         rsu_request_segment_set = set()
            #
            #         if request.content_id in rsu_j.cache_content_segments_set:
            #             for seg_i in range(self.content_library[request.content_id].segment_units):
            #                 if seg_i in rsu_j.cache_content_segments_set[request.content_id]:
            #                     if request.left_segment_list[seg_i] > 0 and not process_flag:
            #                         process_flag = True
            #                         target_node, target_seg = rsu_j, seg_i
            #                         source_vehicle.in_comm_flag = True
            #                 else:
            #                     rsu_j.seg_miss_cnt += 1
            #                     if rsu_j.seg_miss_cnt == p.RSU_DECISION_INTERVAL:
            #                         rsu_j.seg_miss_cnt = 0
            #                         rsu_request_segment_set.add((rsu_j.id, request.content_id, seg_i))
            #         else:
            #             for seg_i in range(self.content_library[request.content_id].segment_units):
            #                 rsu_j.seg_miss_cnt += 1
            #                 if rsu_j.seg_miss_cnt == p.RSU_DECISION_INTERVAL:
            #                     rsu_j.seg_miss_cnt = 0
            #                     rsu_request_segment_set.add((rsu_j.id, request.content_id, seg_i))
            #
            #         if process_flag:
            #             break
            #
            #     self.initialize_rsu_requests(rsu_request_segment_set)

            # D2D通信提供内容
            if process_flag:
                target_node.update_cache_properties(content_id=request.content_id,
                                                    seg_id=target_seg,
                                                    seg_size=self.content_library[request.content_id].segment_unit_size,
                                                    nowTime=self.timer + source_vehicle.in_comm_time_per_slot,
                                                    popularity=self.varying_content_popularity_dist[self.timer][
                                                        request.content_id])

        if not process_flag:  # 最后向MBS请求content_id的内容
            for seg_i in range(len(request.left_segment_list)):
                if request.left_segment_list[seg_i] <= 0:
                    continue
                target_node, target_seg = self.mbs, seg_i
                source_vehicle.in_comm_flag = True
                break

        assert target_node is not None, 'Invalid target node for a new content delivery'
        duration = self.transfer_segment_of_request_to_node(source_vehicle, target_node, request, target_seg, left_time)

        # if cache_enabled:
        #     for rsu_i in neighbor_rsus:
        #         for replace_seg_i in range(self.content_library[request.content_id].segment_units):
        #             self.update_cache_space_on_node(rsu_i, request.content_id, replace_seg_i,
        #                                             self.timer + source_vehicle.in_comm_time_per_slot)

        assert 0 < duration, 'Wrong initiate_new_content_request_delivery func output'
        return min(left_time, duration), target_node

    def transfer_segment_of_request_to_node(self, source_veh, target_node, request, seg, left_time):
        request.last_timeslot_segment_provider = target_node
        dis = self.get_distance(source_veh, target_node)
        transmit_rate = None  # MBps
        if target_node.type == 'Vehicle':
            transmit_rate = p.BANDWIDTH_V2V * math.log2(  # V2V Content Delivery
                1 + p.P_VEH * p.PATHLOSS_CONSTANT * (dis ** (-p.PATHLOSS_EXPONENT)) / p.NOISE_POWER) / 8
        elif target_node.type == 'RSU':
            transmit_rate = p.BANDWIDTH_V2R * math.log2(  # V2R Content Delivery
                1 + p.P_RSU * p.PATHLOSS_CONSTANT * (dis ** (-p.PATHLOSS_EXPONENT)) / p.NOISE_POWER) / 8
        else:
            assert target_node.type == 'MBS', 'invalid service provider'

            transmit_rate = p.BANDWIDTH_V2I * math.log2(  # V2I Content Delivery
                1 + p.P_MBS * p.PATHLOSS_CONSTANT * (dis ** (-p.PATHLOSS_EXPONENT)) / p.NOISE_POWER) / 8

        assert transmit_rate is not None, 'invalid transmit rate'

        processed_seg_amount = transmit_rate * left_time * p.TIME_SLOT
        duration = request.left_segment_list[seg] / (transmit_rate * p.TIME_SLOT)
        request.left_segment_list[seg] = max(0, request.left_segment_list[seg] - processed_seg_amount)

        # segment成功传输
        if request.left_segment_list[seg] == 0:
            # 更新target node上的cache hit cnt
            target_node.increment_cache_hit_size_and_cnt(self.timer, request.content_id, self.content_library[request.content_id].segment_unit_size)

            # 重置通信双方的通信状态
            logging.debug(
                f'Request{request.r_id}: {target_node.type}节点{target_node.id}向Vehicle{source_veh.id}节点成功传输了content{request.content_id}的segment{seg}')
            source_veh.in_comm_flag = False
            if target_node.type == 'Vehicle':
                target_node.in_comm_flag = False

            request.last_timeslot_segment_provider = None
            if target_node.type == 'MBS':
                request.mbs_hit_segment_cnt += 1
            elif target_node.type == 'RSU':
                request.v2r_cache_hit_segment_cnt += 1
            else:
                assert target_node.type == 'Vehicle', 'Invalid target node type'
                request.v2v_cache_hit_segment_cnt += 1

            assert source_veh.in_comm_time_per_slot + duration <= 1, 'should <= 1'

            # cache delegate op
            self.update_cache_space_on_node(self.all_vehicle_data[request.vehicle_id], request.r_id,request.content_id, seg,
                                            self.timer + source_veh.in_comm_time_per_slot + duration)

        return duration

    def initialize_rsu_requests(self, rsu_request_segment_set):
        for rsu_id, content_id, seg_id in rsu_request_segment_set:
            target_rsu_node, target_finish_time = None, float('inf')
            for rsu_j in range(len(self.rsus.keys())):
                if rsu_j == rsu_id:
                    continue

                if content_id in self.rsus[rsu_j].cache_content_segments_set and seg_id in \
                        self.rsus[rsu_j].cache_content_segments_set[content_id]:
                    end_time = self.timer + math.ceil(
                        self.content_library[content_id].segment_unit_size * self.rsus_shortest_path_matrix[rsu_id][
                            rsu_j] / p.BANDWIDTH_R2R)
                    if target_finish_time > end_time:
                        target_finish_time = end_time
                        target_rsu_node = rsu_j
            if target_rsu_node is not None:
                self.processing_rsu_requests[target_finish_time].append(
                    m.RSU_Request(rsu_id, self.timer, target_finish_time, content_id, seg_id))

    def update_cache_space_on_node(self, node, r_id,content_id, seg_id,nowTime):
        # 提取现在的nowtime,popularity,等等信息
        obs = self.get_env_observation()
        obs['req_id'] = r_id
        obs['nowTime'] = nowTime
        obs['popularity'] = self.varying_content_popularity_dist[self.timer][content_id]
        obs['static_popularity'] = self.content_library[content_id].popularity
        obs['seg_size'] = self.content_library[content_id].segment_unit_size
        obs['content_id'] = content_id
        obs['seg_id'] = seg_id
        obs['seg_unit'] = self.content_library[content_id].segment_units


        node.update_cache_segments(obs=obs, test_flag=self.test_flag)

    def get_env_observation(self):
        return {'done': self.timer >= self.r_trace[-1][1] and len(self.processing_requests) == 0}

    def get_neighbours_veh(self, cur_vehicles):

        v_set = list(cur_vehicles.keys())
        cnt = len(v_set)

        neighbor_vehicles = [[] for _ in range(cnt)]
        for i in range(cnt):
            for j in range(i + 1, cnt):
                v_i, v_j = cur_vehicles[v_set[i]], cur_vehicles[v_set[j]]
                dis = math.sqrt((v_i.x - v_j.x) ** 2 + (v_i.y - v_j.y) ** 2)
                dis = round(dis)
                if dis <= p.VEH_RANGE:
                    self.sorted_list[(i, dis)].append(v_set[j])
                    self.sorted_list[(j, dis)].append(v_set[i])

        for i in range(cnt):
            for j in range(p.VEH_RANGE):
                if len(self.sorted_list[(i, j)]) > 0:
                    for vj in self.sorted_list[(i, j)]:
                        neighbor_vehicles[i].append(cur_vehicles[vj])
                    self.sorted_list[(i, j)].clear()
            # self.all_vehicle_data[v_set[i]].update_adjacent_vehicles(neighbor_vehicles[i])
            self.all_vehicle_data[v_set[i]].neighbor_vehicles = neighbor_vehicles[i]

    def get_adjacent_vehicles_for_veh(self, v_id):
        return [self.all_vehicle_data[v] for v in self.neighbor_vehicles_dict[self.timer][v_id]]

    def get_adjacent_vehicles_for_rsu(self, rsu_id):
        return [self.all_vehicle_data[v] for v in self.rsu_neighbor_vehicles_dict[self.timer][rsu_id]]

    def get_adjacent_rsus(self, v_id):
        return [self.rsus[r] for r in self.neighbor_rsus_dict[self.timer][v_id]]

    def get_all_request_process_result(self):
        return self.all_request_data

    @staticmethod
    def get_distance(n_i, n_j):
        assert isinstance(n_i, m.Node) and isinstance(n_j, m.Node), 'Invalid Node Objects'
        return math.sqrt((n_i.x - n_j.x) ** 2 + (n_i.y - n_j.y) ** 2)

    @staticmethod
    def get_vehicle_from_vid(cur_vehicles, v_id):
        assert v_id in cur_vehicles, 'invalid v_id'
        return cur_vehicles[v_id]

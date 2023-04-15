import random
from vanet.tools.models import AbstractCacheModel


class LFU_Cache_Delegate(AbstractCacheModel):
    alg_name = 'LFU'

    def __init__(self, interval=float('inf')):
        super().__init__()
        self.interval = interval
        self.alg_name += f'-{str(interval)}'

    def decision(self, obs, *params):
        newSeg, buffer, left_capacity = obs['newSeg'], obs['buffer'], obs['left_capacity']
        del_list = []

        for id, seg in buffer.items():         # 去除时间差超过S_INTERVAL的访问
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
        buffer[newSeg.id] = newSeg
        replace_flag = True
        return replace_flag, del_list


'''
class GDSF_Cache_Delegate(AbstractCacheModel):
    alg_name = 'GDSF'
    L_value = 0

    def decision(self, newSeg, seg_data, left_ca):
        self.delList = []
        self.replacement(newSeg, seg_data['frequency'])
        return self.delList


    def replacement(self, newSeg, frequency):
        self.delList = []
        while self.capacity < self.totalSize + newSeg.size:
            self.spareBuffer()
        self.buffer[newSeg.id] = SegmentInCache_GDSF(newSeg.id, newSeg.size, frequency, self.L_value)
        self.totalSize += newSeg.size

    def spareBuffer(self):
        tmpSeg = SegmentInCache_GDSF(0, 1, -1, 0)
        for key in self.buffer:
            it = self.buffer[key]
            if tmpSeg.frequency == -1 or tmpSeg.keyValue > it.keyValue:
                tmpSeg = copy.deepcopy(it)
        self.totalSize -= tmpSeg.size
        self.L_value = tmpSeg.keyValue      # 每次替换都要更新L值
        self.delList.append(tmpSeg)
        del self.buffer[tmpSeg.id]
'''


class LRU_Cache_Delegate(AbstractCacheModel):
    alg_name = 'LRU'

    def __init__(self):
        super().__init__()

    def decision(self, obs, *params):
        newSeg, buffer, left_capacity = obs['newSeg'], obs['buffer'], obs['left_capacity']
        del_list = []
        sorted_buffer = sorted(buffer.values(), key=lambda x: x.time)
        for seg in sorted_buffer:
            if left_capacity >= newSeg.size:
                break
            del buffer[seg.id]
            left_capacity += seg.size
            del_list.append(seg)
        buffer[newSeg.id] = newSeg
        return True, del_list


class RC_Cache_Delegate(AbstractCacheModel):
    alg_name = 'RC'

    def __init__(self):
        super().__init__()

    def decision(self, obs, *params):
        newSeg, buffer, left_capacity = obs['newSeg'], obs['buffer'], obs['left_capacity']
        del_list = []
        random_buffer = list(buffer.values())
        random.shuffle(random_buffer)
        for seg in random_buffer:
            if left_capacity >= newSeg.size:
                break
            del buffer[seg.id]
            left_capacity += seg.size
            del_list.append(seg)
        buffer[newSeg.id] = newSeg
        return True, del_list


class GCP_Cache_Delegate(AbstractCacheModel):
    alg_name = 'GCP'

    def __init__(self):
        super().__init__()

    def decision(self, obs, *params):
        newSeg, buffer, left_capacity = obs['newSeg'], obs['buffer'], obs['left_capacity']
        del_list = []
        sorted_buffer = sorted(buffer.values(), key=lambda x: x.popularity)
        for seg in sorted_buffer:
            if left_capacity >= newSeg.size:
                break
            del buffer[seg.id]
            left_capacity += seg.size
            del_list.append(seg)    
        buffer[newSeg.id] = newSeg
        return True, del_list


"""
class SegmentInCache_MobilityAware(Segment):
    def __init__(self, id, size, weight, d2dOffloadingRatio) -> None:
        super().__init__(id, size)
        self.weight = weight
        self.d2dOffloadingRatio = d2dOffloadingRatio


class MobilityAware_Cache_Delegate(AbstractCacheModel):
    alg_name = 'MobilityAware'

    def __init__(self, capacity):
        super().__init__(capacity)
        self.delList = []

    def decision(self, newSeg):
        self.delList = []
        if newSeg.id not in self.buffer:
            self.replacement(newSeg)
        else:
            self.updateWeight(newSeg)
        return self.delList

    def replacement(self, newSeg):
        while self.capacity < self.totalSize + newSeg.size:
            self.spareBuffer()
        self.buffer[newSeg.id] = SegmentInCache_MobilityAware(newSeg.id, newSeg.size, self.calculateWeight(newSeg),
                                                              newSeg.d2dOffloadingRatio)
        self.totalSize += newSeg.size

    def spareBuffer(self):
        sortedSegs = sorted(self.buffer.values(), key=lambda x: x.weight)
        segToRemove = sortedSegs[0]
        self.totalSize -= segToRemove.size
        del self.buffer[segToRemove.id]
        self.delList.append(segToRemove)

    def updateWeight(self, seg):
        segInCache = self.buffer[seg.id]
        segInCache.weight = self.calculateWeight(seg)

    def calculateWeight(self, seg):
        # Calculate the weight of the given segment based on its size and user mobility.
        # The weight is calculated using the formula: weight = size * (1 - D2D_offloading_ratio)
        weight = seg.size * (1 - seg.d2dOffloadingRatio)
        return weight
"""

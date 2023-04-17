import random
import numpy as np
import matplotlib.pyplot as plt


xMin=0;xMax=100
yMin=0;yMax=100
xDelta=xMax-xMin;yDelta=yMax-yMin
areaTotal=xDelta*yDelta
lambda0=20  # 用户数目的均值
device_trace = {}
with open('device_trace.txt', 'w') as f:
    f.write('step 0' + '\n')
    numbPoints = np.random.poisson(lambda0)  # 用户数目
    device_num = np.random.randint(0, lambda0, numbPoints)
    device_num.sort()
    device_num = list(set(device_num))
    xx = []
    yy = []
    for i in range(len(device_num)):
        xx.append(xDelta*np.random.uniform(0,1)+xMin)
        yy.append(yDelta*np.random.uniform(0,1)+yMin)
        plt.xlim([0,100])
        plt.ylim([0,100])
        plt.grid()
        plt.scatter(xx,yy,label='User Equipment')
        plt.scatter(50,50,label = 'Edge Sever')
        plt.legend(loc='best')
        plt.show()
    f.write('step 1' + '\n')
    for id in range(len(device_num)):
        f.write(f'{device_num[id]} {xx[id]} {yy[id]}' + '\n')
    for i in range(2000):
        f.write(f'step {i+2} '+ '\n')
        for t in range(len(device_num)):
            v_x = np.random.uniform(-3, 3)
            v_y = np.random.uniform(-3, 3)
            xx[t] += v_x
            yy[t] += v_y
        for id in range(len(device_num)):
            f.write(f'{device_num[id]} {xx[id]} {yy[id]}' + '\n')
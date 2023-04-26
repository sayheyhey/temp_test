import random
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(3)

xMin=0;xMax=100
yMin=0;yMax=100
xDelta=xMax-xMin;yDelta=yMax-yMin
areaTotal=xDelta*yDelta
lambda0=20  # 用户数目的均值
device_trace = {}
with open('vehicle_trace.txt', 'w') as f:
    f.write('step 0' + '\n')
    # numbPoints = 10  # 用户数目
    # device_num = np.random.randint(0, 20,size=numbPoints)
    # device_num.sort()
    # device_num = list(set(device_num))
    numbPoints = 30
    device_num =[]
    for i in range(numbPoints):
        device_num.append(i+1)

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
    for i in range(244):
        f.write(f'step {i+2} '+ '\n')
        for t in range(len(device_num)):
            v_x = np.random.uniform(-4, 4)
            v_y = np.random.uniform(-4, 4)
            xx[t] += v_x
            yy[t] += v_y
        for id in range(len(device_num)):
            f.write(f'{device_num[id]} {xx[id]} {yy[id]}' + '\n')
import numpy as np
import scipy.io as spio
from dataset import trafficDataLoader

taskID = 1

dataLoader = trafficDataLoader(taskID)
data = dataLoader.data[0, :, :, 0].transpose()  # [n, T]

n, T = data.shape
x = data[:, T - 28*144: T - 7*144]
y = (x[:, :7*144] + x[:, 7*144:14*144] + x[:, 14*144:]) / 3

error = y - data[:, T - 7*144:]
MSE = (error**2).mean()
VD = ((error - error.mean())**2).mean()
print('MSE: %.4f, VD: %.4f' % (MSE, VD))

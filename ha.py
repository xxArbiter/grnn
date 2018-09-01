import numpy as np
import scipy.io as spio
from dataset import trafficDataLoader

taskID = 1

dataLoader = trafficDataLoader(taskID)
data = dataLoader.data  # [n, T]

n, T = data.shape
x = data[:, T - 28*144: T - 7*144]
y = (x[:, :7*144] + x[:, 7*144:14*144] + x[:, 14*144:]) / 3
MSE = ((y - data[:, T - 7*144:])**2).mean()
print(MSE)